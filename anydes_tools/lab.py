from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
import sys
from typing import Callable
from uuid import uuid4

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .audit import ToolAuditLog
from .composition import ToolComposer
from .contracts import (
    CompositionPlan,
    CompositionStep,
    PermissionEffect,
    ToolRequest,
)
from .executor import ToolExecutor
from .path_policy import FilesystemBoundary
from .permissions import PermissionBridge
from .registry import build_official_registry
from .resolver import CapabilityResolver


ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "anydes_tools"
MANUAL = TOOLS / "workspaces" / "manual_v001"


class ExecutionThread(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, operation: Callable[[], object]) -> None:
        super().__init__()
        self.operation = operation

    def run(self) -> None:
        try:
            self.completed.emit(self.operation())
        except BaseException as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ToolsLab(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        MANUAL.mkdir(parents=True, exist_ok=True)
        self.registry = build_official_registry(TOOLS)
        self.boundary = FilesystemBoundary(ROOT, TOOLS)
        self.executor = ToolExecutor(
            ROOT,
            TOOLS,
            self.registry,
            PermissionBridge(),
            ToolAuditLog(TOOLS / "logs" / "manual-audit.jsonl"),
        )
        self.resolver = CapabilityResolver(self.registry, self.boundary)
        self.active_workspace = MANUAL
        self.authorized_scopes: tuple[str, ...] = ()
        self._threads: set[ExecutionThread] = set()

        self.setWindowTitle("Anydes Tools v001 - Staging")
        self.resize(1080, 760)
        self.setCentralWidget(self._build_ui())
        self._refresh_capabilities()

    def _build_ui(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        title = QLabel("Anydes Tool Architecture v001")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        subtitle = QLabel("Laboratorio aislado. Produccion y Core permanecen bloqueados.")
        subtitle.setStyleSheet("color: #555;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        workspace_row = QHBoxLayout()
        self.workspace_label = QLabel(str(self.active_workspace))
        choose = QPushButton("Autorizar carpeta...")
        choose.clicked.connect(self._choose_workspace)
        internal = QPushButton("Usar workspace interno")
        internal.clicked.connect(self._use_internal_workspace)
        workspace_row.addWidget(QLabel("Workspace:"))
        workspace_row.addWidget(self.workspace_label, 1)
        workspace_row.addWidget(choose)
        workspace_row.addWidget(internal)
        layout.addLayout(workspace_row)

        tabs = QTabWidget()
        tabs.addTab(self._files_tab(), "Archivos y documentos")
        tabs.addTab(self._developer_tab(), "Desarrollo")
        tabs.addTab(self._archive_tab(), "ZIP")
        tabs.addTab(self._capabilities_tab(), "Capacidades")
        layout.addWidget(tabs, 1)

        self.status = QLabel("Listo")
        self.status.setStyleSheet("padding: 6px; background: #f1f3f5;")
        layout.addWidget(self.status)
        return root

    def _files_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QFormLayout()
        self.file_name = QLineEdit("nota.txt")
        self.file_content = QPlainTextEdit("Hola desde Anydes Tools v001.")
        self.document_format = QComboBox()
        self.document_format.addItems(["TXT", "Markdown", "DOCX", "PDF"])
        form.addRow("Nombre:", self.file_name)
        form.addRow("Formato:", self.document_format)
        form.addRow("Contenido:", self.file_content)
        layout.addLayout(form)
        row = QHBoxLayout()
        create = QPushButton("Crear documento")
        create.clicked.connect(self._create_document)
        read = QPushButton("Leer archivo")
        read.clicked.connect(self._read_file)
        compose = QPushButton("TXT seleccionado a PDF")
        compose.clicked.connect(self._compose_txt_pdf)
        row.addWidget(create)
        row.addWidget(read)
        row.addWidget(compose)
        layout.addLayout(row)
        self.file_output = QPlainTextEdit()
        self.file_output.setReadOnly(True)
        layout.addWidget(self.file_output, 1)
        return widget

    def _developer_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QFormLayout()
        self.code_name = QLineEdit("ejemplo.py")
        self.code_language = QComboBox()
        self.code_language.addItems(["Python", "C++"])
        self.code_editor = QPlainTextEdit("print('Hola desde el sandbox')\n")
        form.addRow("Archivo:", self.code_name)
        form.addRow("Lenguaje:", self.code_language)
        form.addRow("Codigo:", self.code_editor)
        layout.addLayout(form)
        row = QHBoxLayout()
        save = QPushButton("Guardar codigo")
        save.clicked.connect(self._save_code)
        run = QPushButton("Guardar y ejecutar")
        run.clicked.connect(self._run_code)
        row.addWidget(save)
        row.addWidget(run)
        layout.addLayout(row)
        self.code_output = QPlainTextEdit()
        self.code_output.setReadOnly(True)
        layout.addWidget(self.code_output, 1)
        return widget

    def _archive_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QFormLayout()
        self.archive_sources = QLineEdit("nota.txt")
        self.archive_name = QLineEdit("archivos.zip")
        self.extract_name = QLineEdit("extraido")
        form.addRow("Fuentes (separadas por ;):", self.archive_sources)
        form.addRow("ZIP:", self.archive_name)
        form.addRow("Carpeta de extraccion:", self.extract_name)
        layout.addLayout(form)
        row = QHBoxLayout()
        compress = QPushButton("Comprimir")
        compress.clicked.connect(self._compress)
        inspect = QPushButton("Inspeccionar ZIP")
        inspect.clicked.connect(self._inspect_archive)
        extract = QPushButton("Extraer de forma segura")
        extract.clicked.connect(self._extract)
        row.addWidget(compress)
        row.addWidget(inspect)
        row.addWidget(extract)
        layout.addLayout(row)
        self.archive_output = QPlainTextEdit()
        self.archive_output.setReadOnly(True)
        layout.addWidget(self.archive_output, 1)
        return widget

    def _capabilities_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.capability_list = QListWidget()
        self.health_output = QPlainTextEdit()
        self.health_output.setReadOnly(True)
        refresh = QPushButton("Actualizar estado")
        refresh.clicked.connect(self._refresh_capabilities)
        layout.addWidget(self.capability_list, 1)
        layout.addWidget(refresh)
        layout.addWidget(self.health_output, 1)
        return widget

    def _choose_workspace(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Autorizar workspace")
        if not selected:
            return
        decision = self.boundary.classify(
            Path(selected) / ".scope-check",
            operation="WRITE",
            authorized_scopes=(selected,),
        )
        if not decision.allowed:
            QMessageBox.warning(self, "Carpeta no permitida", decision.reason)
            return
        self.active_workspace = Path(selected).resolve()
        self.authorized_scopes = (str(self.active_workspace),)
        self.workspace_label.setText(str(self.active_workspace))

    def _use_internal_workspace(self) -> None:
        self.active_workspace = MANUAL
        self.authorized_scopes = ()
        self.workspace_label.setText(str(self.active_workspace))

    def _path(self, value: str) -> Path:
        candidate = Path(value.strip())
        return candidate if candidate.is_absolute() else self.active_workspace / candidate

    def _create_document(self) -> None:
        mapping = {
            "TXT": ("document.create_txt", ".txt"),
            "Markdown": ("document.create_markdown", ".md"),
            "DOCX": ("document.create_docx", ".docx"),
            "PDF": ("document.create_pdf", ".pdf"),
        }
        capability, suffix = mapping[self.document_format.currentText()]
        path = self._path(self.file_name.text()).with_suffix(suffix)
        self._execute(
            capability,
            {"path": str(path), "content": self.file_content.toPlainText(), "overwrite": path.exists()},
            self.file_output,
        )

    def _read_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "Leer archivo", str(self.active_workspace))
        if selected:
            self._execute("document.read_document", {"path": selected}, self.file_output)

    def _compose_txt_pdf(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "Seleccionar TXT", str(self.active_workspace), "Text (*.txt *.md)")
        if not selected:
            return
        destination, _ = QFileDialog.getSaveFileName(self, "Guardar PDF", str(self.active_workspace / "resultado.pdf"), "PDF (*.pdf)")
        if not destination:
            return
        plan = CompositionPlan(
            plan_id=f"manual:{uuid4().hex}",
            steps=(
                CompositionStep("read", "filesystem.read", {"path": selected}),
                CompositionStep("pdf", "document.create_pdf", {"path": destination, "content": "$steps.read.content"}),
            ),
        )
        template = self._request("", "", {})
        self._run_async(
            lambda: ToolComposer(self.registry, self.executor).execute(plan, template),
            self.file_output,
        )

    def _save_code(self) -> None:
        path = self._code_path()
        self._execute(
            "developer.write_code",
            {"path": str(path), "code": self.code_editor.toPlainText(), "overwrite": path.exists()},
            self.code_output,
        )

    def _run_code(self) -> None:
        path = self._code_path()
        write = self._request_for_capability(
            "developer.write_code",
            {"path": str(path), "code": self.code_editor.toPlainText(), "overwrite": path.exists()},
            auto_confirm_overwrite=True,
        )
        if write is None:
            return
        run_capability = "developer.run_python" if path.suffix == ".py" else "developer.run_cpp"
        run_request = self._request_for_capability(
            run_capability,
            {"path": str(path)},
            force_confirmed=(PermissionEffect.EXECUTE_CODE,),
        )
        if run_request is None:
            return

        def operation():
            saved = self.executor.execute(write)
            if saved.status.value != "SUCCESS":
                return saved
            return self.executor.execute(run_request)

        self._run_async(operation, self.code_output)

    def _code_path(self) -> Path:
        suffix = ".py" if self.code_language.currentText() == "Python" else ".cpp"
        return self._path(self.code_name.text()).with_suffix(suffix)

    def _compress(self) -> None:
        sources = [str(self._path(item)) for item in self.archive_sources.text().split(";") if item.strip()]
        destination = self._path(self.archive_name.text()).with_suffix(".zip")
        self._execute("archive.compress_zip", {"sources": sources, "destination": str(destination)}, self.archive_output)

    def _inspect_archive(self) -> None:
        self._execute("archive.inspect", {"path": str(self._path(self.archive_name.text()).with_suffix('.zip'))}, self.archive_output)

    def _extract(self) -> None:
        self._execute(
            "archive.extract_zip",
            {"path": str(self._path(self.archive_name.text()).with_suffix('.zip')), "destination": str(self._path(self.extract_name.text()))},
            self.archive_output,
        )

    def _execute(self, capability: str, arguments: dict[str, object], output: QPlainTextEdit) -> None:
        request = self._request_for_capability(capability, arguments, auto_confirm_overwrite=True)
        if request:
            self._run_async(lambda: self.executor.execute(request), output)

    def _request_for_capability(
        self,
        capability: str,
        arguments: dict[str, object],
        *,
        auto_confirm_overwrite: bool = False,
        force_confirmed: tuple[PermissionEffect, ...] = (),
    ) -> ToolRequest | None:
        tool = self.registry.find_by_capability(capability)[0]
        effect = tool.effect_for(capability, arguments)
        confirmed = set(force_confirmed)
        if effect in {PermissionEffect.OVERWRITE, PermissionEffect.MOVE, PermissionEffect.DELETE, PermissionEffect.EXECUTE_CODE}:
            if effect == PermissionEffect.OVERWRITE and not auto_confirm_overwrite:
                return None
            answer = QMessageBox.question(
                self,
                "Confirmar accion",
                f"Permitir una vez: {effect.value}?\n\n{capability}",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return None
            confirmed.add(effect)
        return self._request(tool.manifest.tool_id, capability, arguments, tuple(confirmed))

    def _request(
        self,
        tool_id: str,
        capability: str,
        arguments: dict[str, object],
        confirmed: tuple[PermissionEffect, ...] = (),
    ) -> ToolRequest:
        return ToolRequest(
            request_id=f"manual:{uuid4().hex}",
            turn_id=f"manual-turn:{uuid4().hex}",
            tool_id=tool_id,
            capability=capability,
            arguments=arguments,
            authorized_scopes=self.authorized_scopes,
            confirmed_effects=confirmed,
            reflection_approved=True,
            explicit_user_request=True,
        )

    def _run_async(self, operation: Callable[[], object], output: QPlainTextEdit) -> None:
        self.status.setText("Ejecutando de forma aislada...")
        thread = ExecutionThread(operation)
        self._threads.add(thread)

        def completed(result: object) -> None:
            output.setPlainText(json.dumps(_serializable(result), ensure_ascii=False, indent=2))
            self.status.setText("Operacion terminada")
            self._threads.discard(thread)
            thread.deleteLater()

        def failed(error: str) -> None:
            output.setPlainText(error)
            self.status.setText("La herramienta fallo sin afectar a Anydes")
            self._threads.discard(thread)
            thread.deleteLater()

        thread.completed.connect(completed)
        thread.failed.connect(failed)
        thread.start()

    def _refresh_capabilities(self) -> None:
        self.capability_list.clear()
        self.capability_list.addItems(list(self.registry.list_capabilities(enabled_only=True)))
        self.health_output.setPlainText(json.dumps(self.registry.health_status(), indent=2))


def _serializable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _serializable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = ToolsLab()
    window.show()
    smoke_result = os.getenv("ANYDES_TOOLS_SMOKE_RESULT", "").strip()
    if smoke_result:
        Path(smoke_result).write_text(
            json.dumps(
                {
                    "status": "ready",
                    "tools": len(window.registry.list_tools(enabled_only=True)),
                    "capabilities": len(window.registry.list_capabilities(enabled_only=True)),
                    "workspace": str(window.active_workspace),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    auto_close = int(os.getenv("ANYDES_TOOLS_SMOKE_AUTO_CLOSE_MS", "0") or "0")
    if auto_close > 0:
        QTimer.singleShot(auto_close, app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
