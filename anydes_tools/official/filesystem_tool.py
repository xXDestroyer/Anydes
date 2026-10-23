from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
from typing import Any

from ..context import ToolRuntimeContext
from ..contracts import PermissionEffect, ToolManifest, ToolStatus


class FileSystemTool:
    def __init__(self, manifest: ToolManifest) -> None:
        self.manifest = manifest

    def health(self) -> tuple[ToolStatus, str]:
        return ToolStatus.AVAILABLE, "stdlib_ready"

    def effect_for(self, capability: str, arguments: dict[str, Any]) -> PermissionEffect:
        if capability in {"filesystem.inspect", "filesystem.search", "filesystem.read"}:
            return PermissionEffect.READ
        if capability in {"filesystem.move", "filesystem.rename"}:
            return PermissionEffect.MOVE
        if capability == "filesystem.delete":
            return PermissionEffect.DELETE
        if capability == "filesystem.edit_text":
            return PermissionEffect.EDIT
        if capability == "filesystem.atomic_replace":
            return PermissionEffect(str(arguments.get("commit_effect", PermissionEffect.CREATE.value)))
        if bool(arguments.get("overwrite")):
            return PermissionEffect.OVERWRITE
        return PermissionEffect.CREATE

    def execute(
        self,
        capability: str,
        arguments: dict[str, Any],
        context: ToolRuntimeContext,
    ) -> dict[str, Any]:
        handlers = {
            "filesystem.inspect": self._inspect,
            "filesystem.search": self._search,
            "filesystem.read": self._read,
            "filesystem.create_text": self._create_text,
            "filesystem.edit_text": self._edit_text,
            "filesystem.copy": self._copy,
            "filesystem.move": self._move,
            "filesystem.rename": self._rename,
            "filesystem.mkdir": self._mkdir,
            "filesystem.delete": self._delete,
            "filesystem.atomic_replace": self._atomic_replace,
        }
        return handlers[capability](arguments, context)

    def _inspect(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        path = context.authorize_path(_required(args, "path"), operation="READ")
        if not path.exists():
            raise FileNotFoundError(path)
        stat = path.stat()
        result: dict[str, Any] = {
            "path": str(path),
            "kind": "directory" if path.is_dir() else "file",
            "size": stat.st_size,
            "modified_ns": stat.st_mtime_ns,
        }
        if path.is_file() and stat.st_size <= 64 * 1024 * 1024:
            result["sha256"] = _sha256(path)
        return result

    def _search(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        root = context.authorize_path(_required(args, "root"), operation="READ")
        pattern = str(args.get("pattern") or "*")
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError("unsafe_search_pattern")
        limit = min(max(int(args.get("limit", 200)), 1), 2000)
        matches: list[str] = []
        for item in root.rglob(pattern):
            try:
                safe = context.authorize_path(item, operation="READ")
            except Exception:
                continue
            matches.append(str(safe))
            if len(matches) >= limit:
                break
        return {"root": str(root), "pattern": pattern, "matches": matches, "truncated": len(matches) >= limit}

    def _read(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        path = context.authorize_path(_required(args, "path"), operation="READ")
        max_bytes = min(max(int(args.get("max_bytes", 1_000_000)), 1), 4_000_000)
        data = context.revalidate(path, "READ").read_bytes()
        if len(data) > max_bytes:
            raise ValueError("file_exceeds_read_limit")
        return {"path": str(path), "content": data.decode("utf-8", errors="replace"), "size": len(data)}

    def _create_text(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        path = context.authorize_path(_required(args, "path"), operation="WRITE")
        overwrite = bool(args.get("overwrite", False))
        existed = path.exists()
        if existed and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = str(args.get("content", ""))
        with context.revalidate(path, "WRITE").open("w" if overwrite else "x", encoding="utf-8") as handle:
            handle.write(content)
        effect = PermissionEffect.OVERWRITE if existed else PermissionEffect.CREATE
        return _with_effect({"path": str(path), "bytes": path.stat().st_size}, effect, path)

    def _edit_text(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        path = context.authorize_path(_required(args, "path"), operation="WRITE")
        original = context.revalidate(path, "READ").read_text(encoding="utf-8")
        old = str(_required(args, "old_text"))
        new = str(args.get("new_text", ""))
        count = int(args.get("count", 1))
        if old not in original:
            raise ValueError("edit_anchor_not_found")
        updated = original.replace(old, new, count if count > 0 else -1)
        context.revalidate(path, "WRITE").write_text(updated, encoding="utf-8")
        return _with_effect({"path": str(path), "replacements": original.count(old) if count <= 0 else min(original.count(old), count)}, PermissionEffect.EDIT, path)

    def _copy(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        source = context.authorize_path(_required(args, "source"), operation="READ")
        destination = context.authorize_path(_required(args, "destination"), operation="WRITE")
        overwrite = bool(args.get("overwrite", False))
        existed = destination.exists()
        if existed and not overwrite:
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        context.validate_tree(source, "READ")
        context.revalidate(destination, "WRITE")
        def guarded_copy(src, dst):
            src = context.revalidate(Path(src), "READ")
            dst = context.revalidate(Path(dst), "WRITE")
            with src.open("rb") as reader, dst.open("wb" if overwrite else "xb") as writer:
                shutil.copyfileobj(reader, writer)
            return str(dst)
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=overwrite, copy_function=guarded_copy)
        else:
            guarded_copy(source, destination)
        effect = PermissionEffect.OVERWRITE if existed else PermissionEffect.CREATE
        return _with_effect({"source": str(source), "destination": str(destination)}, effect, destination)

    def _move(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        source = context.authorize_path(_required(args, "source"), operation="WRITE")
        destination = context.authorize_path(_required(args, "destination"), operation="WRITE")
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        context.validate_tree(source, "WRITE")
        context.revalidate(destination, "WRITE")
        if source.drive.casefold() != destination.drive.casefold():
            from ..errors import PathDeniedError
            raise PathDeniedError("DENIED_CROSS_DRIVE_MOVE")
        moved = source.rename(destination)
        result = {"source": str(source), "destination": str(moved)}
        result["_side_effects"] = [
            {"effect": PermissionEffect.MOVE.value, "path": str(source)},
            {"effect": PermissionEffect.CREATE.value, "path": str(moved)},
        ]
        return result

    def _rename(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        source = context.authorize_path(_required(args, "path"), operation="WRITE")
        name = str(_required(args, "new_name"))
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("invalid_new_name")
        destination = context.authorize_path(source.with_name(name), operation="WRITE")
        if destination.exists():
            raise FileExistsError(destination)
        context.validate_tree(source, "WRITE")
        context.revalidate(destination, "WRITE")
        source.rename(destination)
        result = {"source": str(source), "destination": str(destination)}
        result["_side_effects"] = [
            {"effect": PermissionEffect.MOVE.value, "path": str(source)},
            {"effect": PermissionEffect.CREATE.value, "path": str(destination)},
        ]
        return result

    def _mkdir(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        path = context.authorize_path(_required(args, "path"), operation="WRITE")
        if path.exists():
            raise FileExistsError(path)
        context.revalidate(path, "WRITE").mkdir(parents=bool(args.get("parents", True)))
        return _with_effect({"path": str(path)}, PermissionEffect.CREATE, path)

    def _delete(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        path = context.authorize_path(_required(args, "path"), operation="WRITE")
        if not path.exists():
            raise FileNotFoundError(path)
        context.validate_tree(path, "WRITE")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return _with_effect({"path": str(path), "deleted": True}, PermissionEffect.DELETE, path)

    def _atomic_replace(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        source = context.authorize_path(_required(args, "source"), operation="WRITE")
        destination = context.authorize_path(_required(args, "destination"), operation="WRITE")
        if not source.is_file():
            raise FileNotFoundError(source)
        effect = PermissionEffect(str(args.get("commit_effect", PermissionEffect.CREATE.value)))
        existed = destination.exists()
        if existed and effect not in {PermissionEffect.EDIT, PermissionEffect.OVERWRITE}:
            raise FileExistsError(destination)
        if not existed and effect in {PermissionEffect.EDIT, PermissionEffect.OVERWRITE}:
            raise FileNotFoundError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        context.revalidate(source, "WRITE")
        context.revalidate(destination, "WRITE")
        expected = args.get("expected_destination_hash")
        if expected is not None:
            actual = "ABSENT"
            if destination.is_file():
                digest = hashlib.sha256()
                with context.revalidate(destination, "READ").open("rb") as reader:
                    for block in iter(lambda: reader.read(1024 * 1024), b""):
                        digest.update(block)
                actual = digest.hexdigest()
            if actual != expected:
                raise ValueError("STALE_ARTIFACT")
        if effect == PermissionEffect.CREATE:
            # Windows rename fails when the destination already exists.
            if os.name == "nt":
                source.rename(destination)
            else:
                os.link(source, destination)
                source.unlink()
        else:
            os.replace(source, destination)
        return _with_effect(
            {
                "source": str(source),
                "destination": str(destination),
                "atomic": True,
                "bytes": destination.stat().st_size,
            },
            effect,
            destination,
        )


def _required(args: dict[str, Any], key: str) -> Any:
    if key not in args or args[key] in {None, ""}:
        raise ValueError(f"missing_argument:{key}")
    return args[key]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _with_effect(result: dict[str, Any], effect: PermissionEffect, path: Path) -> dict[str, Any]:
    result["_side_effects"] = [{"effect": effect.value, "path": str(path)}]
    return result
