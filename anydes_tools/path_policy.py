from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PureWindowsPath
import stat
from .errors import PathDeniedError


@dataclass(frozen=True)
class PathDecision:
    allowed: bool
    category: str
    path: Path
    reason: str


class FilesystemBoundary:
    """Default deny for implementation; only explicit data workspaces are public."""

    def __init__(self, project_root: Path, tools_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.tools_root = tools_root.resolve()
        self.workspace_root = self.tools_root / "workspaces"
        self.sandbox_root = self.tools_root / "sandbox"
        self.protected_roots = (self.project_root,)
        self.protected_files = (self.project_root / ".env", self.project_root / "Abrir Anydes.bat")

    def authorize(self, raw_path, *, operation, authorized_scopes=(), developer_only=False):
        decision = self.classify(raw_path, operation=operation,
            authorized_scopes=authorized_scopes, developer_only=developer_only)
        if not decision.allowed:
            raise PathDeniedError(decision.reason)
        return decision.path

    def classify(self, raw_path, *, operation, authorized_scopes=(), developer_only=False):
        candidate = self._candidate(raw_path)
        if operation not in {"READ", "WRITE"}:
            raise PathDeniedError("DENIED_UNKNOWN_PATH_OPERATION")
        if self._contains_reparse_point(candidate):
            return PathDecision(False, "DENIED", candidate, "DENIED_REPARSE_POINT")
        canonical = self._canonical(candidate)
        if self._protected(canonical):
            return PathDecision(False, "PROTECTED_CORE", canonical, "DENIED_CORE_PROTECTED")
        scopes = [self.workspace_root]
        for raw in authorized_scopes:
            scope = self._candidate(raw)
            if self._contains_reparse_point(scope) or not self._scope_allowed(scope):
                raise PathDeniedError("DENIED_INVALID_SCOPE")
            scopes.append(self._canonical(scope))
        matched = next((scope for scope in scopes if _inside(canonical, scope)), None)
        if matched is None:
            return PathDecision(False, "DENIED", canonical, "DENIED_OUTSIDE_AUTHORIZED_SCOPE")
        if operation == "WRITE" and canonical in scopes:
            return PathDecision(False, "DENIED", canonical, "DENIED_SCOPE_ROOT_WRITE")
        try:
            if canonical.is_file() and canonical.stat().st_nlink > 1:
                raise PathDeniedError("DENIED_HARDLINK")
        except OSError as exc:
            raise PathDeniedError("DENIED_UNVERIFIABLE_PATH") from exc
        category = "ANYDES_TOOLS" if _inside(canonical, self.workspace_root) else "AUTHORIZED_USER_WORKSPACE"
        return PathDecision(True, category, canonical, "AUTHORIZED")

    def _candidate(self, raw_path):
        text = str(raw_path)
        windows = PureWindowsPath(text)
        if not text or any(ord(ch) < 32 for ch in text):
            raise PathDeniedError("DENIED_INVALID_PATH")
        if text.startswith(("\\\\", "//")) or (windows.drive and not windows.root):
            raise PathDeniedError("DENIED_PATH_NAMESPACE")
        if ".." in windows.parts or ".." in Path(text).parts:
            raise PathDeniedError("DENIED_PATH_TRAVERSAL")
        for part in windows.parts[1:] if windows.anchor else windows.parts:
            stem = part.split(".")[0].upper()
            reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
            if part.endswith((" ", ".")) or ":" in part or any(c in part for c in '<>"|?*') or stem in reserved:
                raise PathDeniedError("DENIED_PATH_NAMESPACE")
        path = Path(text)
        if path.anchor and not path.is_absolute():
            raise PathDeniedError("DENIED_PATH_NAMESPACE")
        return path if path.is_absolute() else self.workspace_root / path

    def _canonical(self, path):
        try:
            return path.resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise PathDeniedError("DENIED_UNRESOLVABLE_PATH") from exc

    def _protected(self, path):
        if any(part.casefold() == ".env" or part.casefold().startswith(".env.") for part in path.parts):
            return True
        return _inside(path, self.project_root) and not _inside(path, self.workspace_root)

    def _scope_allowed(self, path):
        path = self._canonical(path)
        if path == Path(path.anchor) or _inside(self.project_root, path):
            return False
        for value in (os.environ.get("SystemRoot"), os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
            if value and _inside(path, self._canonical(Path(value))):
                return False
        return not self._protected(path)

    def _contains_reparse_point(self, path):
        for item in (path, *path.parents):
            try:
                info = item.lstat()
            except FileNotFoundError:
                continue
            except OSError:
                return True
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                return True
        return False

    def runtime_directory(self, raw_path, kind):
        """Trusted worker allocation, never an authorization scope supplied by a tool."""
        path = self._candidate(raw_path)
        parent = self.sandbox_root / kind
        if kind not in {"runtime_home", "runtime_temp"} or path.parent != parent or self._contains_reparse_point(path):
            raise PathDeniedError("DENIED_RUNTIME_DIRECTORY")
        if not path.is_dir():
            raise PathDeniedError("DENIED_RUNTIME_DIRECTORY")
        return self._canonical(path)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
