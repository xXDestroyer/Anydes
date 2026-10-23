from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .path_policy import FilesystemBoundary


@dataclass(frozen=True)
class ToolRuntimeContext:
    project_root: Path
    tools_root: Path
    boundary: FilesystemBoundary
    authorized_scopes: tuple[str, ...]
    runtime_home: Path
    runtime_temp: Path

    def authorize_path(
        self,
        raw_path: str | Path,
        *,
        operation: str,
        developer_only: bool = False,
    ) -> Path:
        return self.boundary.authorize(
            raw_path,
            operation=operation,
            authorized_scopes=self.authorized_scopes,
            developer_only=developer_only,
        )

    def revalidate(self, path: Path, operation: str) -> Path:
        """Run the boundary again at the IO site, not only during planning."""
        return self.authorize_path(path, operation=operation)

    def validate_tree(self, path: Path, operation: str) -> None:
        self.revalidate(path, operation)
        if path.is_dir():
            import os
            for root, directories, files in os.walk(path, followlinks=False):
                for name in directories + files:
                    self.revalidate(Path(root) / name, operation)
