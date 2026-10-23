from __future__ import annotations

import os
import json
from pathlib import Path
from types import SimpleNamespace
import shutil
import tempfile
import unittest
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from anydes_tools.conversation.controller import ToolConversationController
from anydes_tools.settings.runtime_patch import make_settings_managed_panel
from anydes_tools.settings.store import SettingsStore, WorkspaceValidationError


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "anydes_tools"


class SettingsCapabilityV0015Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = TOOLS / "workspaces" / f"test_settings_{uuid4().hex[:10]}"
        self.root.mkdir(parents=True)
        self.config = self.root / "settings.json"
        self.settings = SettingsStore(ROOT, TOOLS, settings_path=self.config)
        self.controller = ToolConversationController(ROOT, TOOLS, settings=self.settings)
        self.turn = 0

    def tearDown(self) -> None:
        self.controller.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def ask(self, text: str):
        self.turn += 1
        return self.controller.handle(text, turn_id=str(self.turn))

    def test_01_can_create_files_uses_registry_truth(self):
        result = self.ask("¿puedes crear archivos?")
        self.assertTrue(result.handled)
        self.assertIn("sí, puedo crear archivos", result.text.casefold())
        workspace = self.settings.snapshot().active_workspace
        self.assertIn(Path(workspace).name, result.text)
        self.assertNotIn(workspace, result.text)
        self.assertEqual(result.debug["source"], "ToolRegistry+SettingsStore")

    def test_02_tools_question_uses_live_snapshot(self):
        result = self.ask("¿qué herramientas tienes?")
        snapshot = self.controller.registry.capability_snapshot()
        self.assertIn(str(len(snapshot.tools)), result.text)
        self.assertIn(str(len(snapshot.available_capabilities)), result.text)
        self.assertIn("Document", result.text)

    def test_03_workspace_question_reports_internal_scope(self):
        result = self.ask("entonces ¿en qué carpeta tienes permiso?")
        snapshot = self.settings.snapshot()
        self.assertEqual(snapshot.active_workspace_kind, "internal")
        self.assertIn(Path(snapshot.internal_workspace).name, result.text)
        self.assertNotIn(snapshot.internal_workspace, result.text)
        self.assertIn("núcleo", result.text.casefold())

    def test_04_core_is_always_protected(self):
        result = self.ask("¿puedes modificar tu Core?")
        self.assertIn("no", result.text.casefold())
        self.assertIn("protegido", result.text.casefold())
        self.assertFalse(result.executed)

    def test_05_permanent_workspace_persists(self):
        permanent = self.root / "Carpeta Permanente"
        permanent.mkdir()
        snapshot = self.settings.set_permanent_workspace(permanent)
        self.assertEqual(snapshot.active_workspace, str(permanent.resolve()))
        restarted = SettingsStore(ROOT, TOOLS, settings_path=self.config)
        self.assertEqual(restarted.snapshot().permanent_workspace, str(permanent.resolve()))
        self.assertEqual(restarted.snapshot().active_workspace_kind, "permanent")

    def test_06_temporary_workspace_wins_and_expires_on_restart(self):
        permanent = self.root / "Permanente"
        temporary = self.root / "Temporal"
        permanent.mkdir(); temporary.mkdir()
        self.settings.set_permanent_workspace(permanent)
        live = self.settings.set_temporary_workspace(temporary)
        self.assertEqual(live.active_workspace, str(temporary.resolve()))
        self.assertEqual(live.active_workspace_kind, "temporary")
        restarted = SettingsStore(ROOT, TOOLS, settings_path=self.config)
        self.assertEqual(restarted.snapshot().active_workspace, str(permanent.resolve()))
        self.assertFalse(restarted.snapshot().temporary_workspace)

    def test_07_clear_permanent_falls_back_to_internal(self):
        permanent = self.root / "Permanente"
        permanent.mkdir()
        self.settings.set_permanent_workspace(permanent)
        snapshot = self.settings.clear_permanent_workspace()
        self.assertEqual(snapshot.active_workspace, snapshot.internal_workspace)
        self.assertEqual(snapshot.active_workspace_kind, "internal")

    def test_08_folder_with_spaces_is_valid(self):
        workspace = self.root / "Proyecto con espacios"
        workspace.mkdir()
        snapshot = self.settings.set_temporary_workspace(workspace)
        self.assertEqual(snapshot.temporary_workspace, str(workspace.resolve()))

    def test_09_invalid_and_protected_paths_are_denied(self):
        with self.assertRaises(WorkspaceValidationError):
            self.settings.set_permanent_workspace(self.root / "No existe")
        with self.assertRaises(WorkspaceValidationError) as caught:
            self.settings.set_temporary_workspace(ROOT / "programa_principal")
        self.assertIn("PROTECTED", str(caught.exception))

    def test_10_settings_change_updates_conversation_immediately(self):
        workspace = self.root / "Nuevo workspace"
        workspace.mkdir()
        self.settings.set_temporary_workspace(workspace)
        self.assertEqual(self.controller.state.active_workspace, str(workspace.resolve()))
        self.assertIn(str(workspace.resolve()), self.controller.state.authorized_scopes)
        answer = self.ask("¿dónde puedes trabajar?")
        self.assertIn(workspace.name, answer.text)
        self.assertNotIn(str(workspace.resolve()), answer.text)

    def test_11_capability_context_is_dynamic_and_secret_free(self):
        old = os.environ.get("GROQ_API_KEY")
        os.environ["GROQ_API_KEY"] = "gsk-never-render-this-secret"
        try:
            note = self.controller.capability_context_note()
        finally:
            if old is None:
                os.environ.pop("GROQ_API_KEY", None)
            else:
                os.environ["GROQ_API_KEY"] = old
        self.assertIn("tool_system_enabled", note)
        self.assertIn("protected_core", note)
        self.assertNotIn("gsk-never-render", note)

    def test_12_session_scope_is_not_persisted(self):
        scope = self.root / "Scope adicional"
        scope.mkdir()
        self.settings.authorize_session_scope(scope)
        self.assertIn(str(scope.resolve()), self.settings.snapshot().authorized_external_scopes)
        restarted = SettingsStore(ROOT, TOOLS, settings_path=self.config)
        self.assertNotIn(str(scope.resolve()), restarted.snapshot().authorized_external_scopes)

    def test_13_audio_panel_startup_show_is_suppressed(self):
        class Panel:
            def __init__(self):
                self.native_show_count = 0
                self.raised = 0
                self.activated = 0

            def show(self):
                self.native_show_count += 1

            def raise_(self):
                self.raised += 1

            def activateWindow(self):
                self.activated += 1

        Managed = make_settings_managed_panel(Panel)
        panel = Managed()
        panel.show()
        self.assertEqual(panel.native_show_count, 0)
        self.assertTrue(panel.startup_show_suppressed)
        panel.show_from_settings()
        self.assertEqual(panel.native_show_count, 1)
        self.assertEqual(panel.settings_show_count, 1)

    def test_14_settings_dialog_opens_audio_and_contains_no_secrets(self):
        try:
            from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QPushButton, QWidget
            from anydes_tools.settings.ui import AnydesSettingsDialog
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))
        app = QApplication.instance() or QApplication([])

        class AudioPanel:
            def __init__(self):
                self.device = QComboBox(); self.device.addItem("Micrófono de prueba")
                self.vad = QCheckBox(); self.ptt = QPushButton(); self.ptt.setEnabled(True)
                self.pre_roll = QComboBox(); self.pre_roll.addItem("400", 400)
                self.engine = SimpleNamespace(recognizer=None)
                self.stream_state = "Cerrado"
                self.opened = 0

            def show_from_settings(self):
                self.opened += 1

        owner = QWidget()
        owner.input_panel = AudioPanel()
        owner.voice_manager = SimpleNamespace(_warmup_thread=None, engine=None)
        owner.avatar = object()
        owner.vision_session = object()
        owner.assistant = SimpleNamespace(
            config=SimpleNamespace(vision_enabled=True),
            _realtime_service=SimpleNamespace(
                session=SimpleNamespace(continuity_v2=object())
            ),
        )
        dialog = AnydesSettingsDialog(owner, self.controller, self.settings)
        dialog.show(); app.processEvents()
        self.assertTrue(dialog.isVisible())
        dialog.open_audio_diagnostics()
        self.assertEqual(owner.input_panel.opened, 1)
        rendered = "\n".join(label.text() for label in dialog.findChildren(type(dialog._labels["general_tools"])))
        self.assertNotIn("GROQ_API_KEY", rendered)
        self.assertNotIn("gsk-", rendered)
        dialog.hide(); app.processEvents()
        self.assertFalse(dialog.isVisible())
        dialog.deleteLater(); owner.deleteLater()

    def test_15_persistent_file_contains_only_allowlisted_state(self):
        permanent = self.root / "Persistente"
        temporary = self.root / "Solo sesion"
        extra = self.root / "Scope extra"
        permanent.mkdir(); temporary.mkdir(); extra.mkdir()
        self.settings.set_permanent_workspace(permanent)
        self.settings.set_temporary_workspace(temporary)
        self.settings.authorize_session_scope(extra)
        payload = json.loads(self.config.read_text(encoding="utf-8"))
        self.assertEqual(
            set(payload),
            {"schema", "permanent_workspace"},
        )
        self.assertEqual(payload["permanent_workspace"], str(permanent.resolve()))
        rendered = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn(str(temporary.resolve()), rendered)
        self.assertNotIn(str(extra.resolve()), rendered)

    def test_16_settings_ui_reflects_registry_and_active_workspace(self):
        try:
            from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QPushButton, QWidget
            from anydes_tools.settings.ui import AnydesSettingsDialog
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))
        app = QApplication.instance() or QApplication([])
        workspace = self.root / "Workspace visible"
        workspace.mkdir()
        self.settings.set_temporary_workspace(workspace)

        owner = QWidget()
        owner.input_panel = SimpleNamespace(
            device=QComboBox(),
            vad=QCheckBox(),
            ptt=QPushButton(),
            pre_roll=QComboBox(),
            engine=SimpleNamespace(recognizer=None),
            stream_state="Cerrado",
        )
        owner.input_panel.device.addItem("Micrófono real")
        owner.input_panel.pre_roll.addItem("400", 400)
        owner.voice_manager = SimpleNamespace(_warmup_thread=None, engine=None)
        owner.avatar = object()
        owner.vision_session = object()
        owner.assistant = SimpleNamespace(
            config=SimpleNamespace(vision_enabled=True),
            _realtime_service=SimpleNamespace(session=SimpleNamespace(continuity_v2=object())),
        )
        dialog = AnydesSettingsDialog(owner, self.controller, self.settings)
        dialog.refresh(); app.processEvents()
        snapshot = self.controller.registry.capability_snapshot()
        self.assertEqual(dialog._labels["tools_count"].text(), str(len(snapshot.tools)))
        self.assertEqual(
            dialog._labels["capabilities_count"].text(),
            str(len(snapshot.available_capabilities)),
        )
        self.assertEqual(dialog._labels["tools_workspace"].text(), str(workspace.resolve()))
        self.assertEqual(dialog._labels["general_protection"].text(), "ACTIVADA")
        dialog.deleteLater(); owner.deleteLater()

    def test_17_gear_button_opens_settings_callback(self):
        try:
            from PySide6.QtWidgets import QApplication, QWidget
            from anydes_tools.settings.ui import build_settings_button
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))
        app = QApplication.instance() or QApplication([])
        parent = QWidget()
        opened = []
        button = build_settings_button(parent, lambda: opened.append(True))
        button.click(); app.processEvents()
        self.assertEqual(button.text(), "⚙")
        self.assertEqual(button.toolTip(), "Configuración")
        self.assertEqual(opened, [True])
        button.deleteLater(); parent.deleteLater()

    def test_18_generic_word_capability_question_does_not_create_pending_action(self):
        from anydes_tools.conversation.interaction_router import (
            InteractionCategory,
            InteractionDecision,
            InteractionRouter,
        )

        self.controller.interaction_router = InteractionRouter(
            lambda request: InteractionDecision(
                InteractionCategory.CAPABILITY_QUERY,
                1.0,
                primary_goal="describe current document capability",
            )
        )
        result = self.ask("¿puedes crear Word?")
        self.assertTrue(result.handled)
        self.assertFalse(result.executed)
        self.assertIsNone(self.controller.state.pending_tool_action)
        self.assertIn("sí, puedo crear documentos word", result.text.casefold())


if __name__ == "__main__":
    unittest.main()
