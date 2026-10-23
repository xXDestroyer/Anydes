from __future__ import annotations

from pathlib import Path, PurePosixPath
import stat
from typing import Any
import zipfile

from ..context import ToolRuntimeContext
from ..contracts import PermissionEffect, ToolManifest, ToolStatus


class ArchiveTool:
    def __init__(self, manifest: ToolManifest) -> None:
        self.manifest = manifest

    def health(self) -> tuple[ToolStatus, str]:
        return ToolStatus.AVAILABLE, "stdlib_zip_ready"

    def effect_for(self, capability: str, arguments: dict[str, Any]) -> PermissionEffect:
        if capability == "archive.inspect":
            return PermissionEffect.READ
        if bool(arguments.get("overwrite")):
            return PermissionEffect.OVERWRITE
        return PermissionEffect.CREATE

    def execute(self, capability: str, arguments: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        if capability == "archive.inspect":
            return self._inspect(arguments, context)
        if capability == "archive.compress_zip":
            return self._compress(arguments, context)
        return self._extract(arguments, context)

    def _inspect(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        path = context.authorize_path(_required(args, "path"), operation="READ")
        with zipfile.ZipFile(context.revalidate(path, "READ")) as archive:
            entries = [_entry(item) for item in archive.infolist()[:2000]]
        return {"path": str(path), "entries": entries, "safe": all(item["safe"] for item in entries)}

    def _compress(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        destination = context.authorize_path(_required(args, "destination"), operation="WRITE")
        if destination.suffix.casefold() != ".zip":
            raise ValueError("destination_must_be_zip")
        existed = destination.exists()
        if existed and not bool(args.get("overwrite", False)):
            raise FileExistsError(destination)
        raw_sources = args.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError("sources_must_be_nonempty_array")
        sources = [context.authorize_path(str(item), operation="READ") for item in raw_sources]
        destination.parent.mkdir(parents=True, exist_ok=True)
        for source in sources:
            context.validate_tree(source, "READ")
        with zipfile.ZipFile(context.revalidate(destination, "WRITE"), "w" if bool(args.get("overwrite")) else "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for source in sources:
                if source.is_dir():
                    for item in source.rglob("*"):
                        safe = context.authorize_path(item, operation="READ")
                        if safe.is_file():
                            archive.write(safe, arcname=f"{source.name}/{safe.relative_to(source).as_posix()}")
                else:
                    archive.write(context.revalidate(source, "READ"), arcname=source.name)
        effect = PermissionEffect.OVERWRITE if existed else PermissionEffect.CREATE
        return _effect({"path": str(destination), "bytes": destination.stat().st_size}, effect, destination)

    def _extract(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        source = context.authorize_path(_required(args, "path"), operation="READ")
        destination = context.authorize_path(_required(args, "destination"), operation="WRITE")
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError("extraction_destination_not_empty")
        extracted: list[str] = []
        with zipfile.ZipFile(context.revalidate(source, "READ")) as archive:
            entries = archive.infolist()
            if len(entries) > 5000:
                raise ValueError("archive_entry_limit_exceeded")
            if sum(info.file_size for info in entries) > 1024 * 1024 * 1024:
                raise ValueError("archive_uncompressed_size_limit_exceeded")
            if any(not _safe_info(info) for info in entries):
                unsafe = next(info.filename for info in entries if not _safe_info(info))
                raise ValueError(f"unsafe_zip_entry:{unsafe}")
            targets = [
                context.authorize_path(destination / PurePosixPath(info.filename), operation="WRITE")
                for info in entries
            ]
            destination.mkdir(parents=True, exist_ok=True)
            for info, target in zip(entries, targets):
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as reader, context.revalidate(target, "WRITE").open("xb") as writer:
                    while block := reader.read(1024 * 1024):
                        writer.write(block)
                extracted.append(str(target))
        result = {"source": str(source), "destination": str(destination), "files": extracted}
        result["_side_effects"] = [{"effect": PermissionEffect.CREATE.value, "path": str(destination)}]
        return result


def _entry(info: zipfile.ZipInfo) -> dict[str, Any]:
    return {"name": info.filename, "size": info.file_size, "safe": _safe_info(info)}


def _safe_info(info: zipfile.ZipInfo) -> bool:
    path = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in info.filename
        and ":" not in info.filename
        and not str(path).startswith(("/", "\\"))
        and not stat.S_ISLNK(mode)
    )


def _required(args: dict[str, Any], key: str) -> Any:
    if key not in args or args[key] in {None, ""}:
        raise ValueError(f"missing_argument:{key}")
    return args[key]


def _effect(result: dict[str, Any], effect: PermissionEffect, path: Path) -> dict[str, Any]:
    result["_side_effects"] = [{"effect": effect.value, "path": str(path)}]
    return result
