from __future__ import annotations

import os
from pathlib import Path

from .store import settings_for


def make_settings_managed_panel(panel_factory):
    """Suppress only the startup show; explicit Settings opening remains available."""

    class SettingsManagedVoicePanel(panel_factory):
        startup_show_suppressed = False
        settings_show_count = 0

        def show(self):
            self.startup_show_suppressed = True
            return None

        def show_from_settings(self):
            self.settings_show_count += 1
            panel_factory.show(self)
            raise_window = getattr(self, "raise_", None)
            if callable(raise_window):
                raise_window()
            activate = getattr(self, "activateWindow", None)
            if callable(activate):
                activate()

    SettingsManagedVoicePanel.__name__ = f"SettingsManaged{panel_factory.__name__}"
    return SettingsManagedVoicePanel


def install_settings_window(module, project_root: Path, *, performance: bool = False) -> None:
    from ..conversation.runtime_patch import ensure_controller_for
    from PySide6.QtCore import QTimer
    from .ui import AnydesSettingsDialog, build_settings_button

    class_name = "PerformanceCompanionWindow" if performance else "AnydesPySideWindow"
    base = getattr(module, class_name)
    root = project_root.resolve()

    class SettingsEnabledWindow(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tool_conversation_controller = ensure_controller_for(self.assistant, root)
            self.anydes_settings = settings_for(root, root / "anydes_tools")
            self.settings_dialog = AnydesSettingsDialog(
                self,
                self.tool_conversation_controller,
                self.anydes_settings,
            )
            self.settings_button = build_settings_button(self, self.open_settings)
            self._place_settings_button(performance)
            self.tool_conversation_controller.audit.record(
                "settings_ui_ready",
                audio_panel_visible=bool(getattr(self.input_panel, "isVisible", lambda: False)()),
                active_workspace=self.anydes_settings.snapshot().active_workspace,
            )
            if os.getenv("ANYDES_SETTINGS_V001_5_SMOKE", "").strip().casefold() in {
                "1", "true", "yes", "on",
            }:
                QTimer.singleShot(1800, self._open_settings_smoke)

        def _place_settings_button(self, is_performance: bool) -> None:
            if is_performance:
                controls = getattr(getattr(self, "chat_window", None), "controls", None)
                if isinstance(controls, list):
                    controls.append(self.settings_button)
                    layout = getattr(getattr(self, "chat_window", None), "title_layout", None)
                    if layout is not None:
                        layout.addWidget(self.settings_button)
                        return
                self.settings_button.setParent(self)
                self.settings_button.move(max(8, self.width() - 48), 8)
                self.settings_button.show()
                return
            target = getattr(self, "inner_button", None)
            found = _find_layout_with_widget(self.centralWidget().layout(), target)
            if found is not None:
                layout, index = found
                layout.insertWidget(index, self.settings_button)
            else:
                self.settings_button.setParent(self.centralWidget())
                self.settings_button.move(max(8, self.width() - 56), 14)
                self.settings_button.show()

        def open_settings(self):
            self.settings_dialog.refresh()
            self.settings_dialog.show()
            self.settings_dialog.raise_()
            self.settings_dialog.activateWindow()
            self.tool_conversation_controller.audit.record(
                "settings_opened",
                page=self.settings_dialog.navigation.currentItem().text(),
            )

        def _open_settings_smoke(self):
            self.open_settings()
            output = root / "anydes_tools" / "logs" / "settings-panel-v001_5.png"
            saved = self.settings_dialog.grab().save(str(output), "PNG")
            self.tool_conversation_controller.audit.record(
                "settings_smoke_capture",
                saved=bool(saved),
                path=str(output),
                audio_panel_visible=bool(getattr(self.input_panel, "isVisible", lambda: False)()),
            )
            self.tool_conversation_controller.audit.record(
                "settings_smoke_passed",
                dialog_visible=bool(self.settings_dialog.isVisible()),
                audio_panel_visible=bool(getattr(self.input_panel, "isVisible", lambda: False)()),
                registry_tools=len(self.tool_conversation_controller.registry.list_tools()),
                active_workspace=self.anydes_settings.snapshot().active_workspace,
            )

        def closeEvent(self, event):
            self.settings_dialog.hide()
            super().closeEvent(event)

    SettingsEnabledWindow.__name__ = f"SettingsEnabled{base.__name__}"
    setattr(module, class_name, SettingsEnabledWindow)


def _find_layout_with_widget(layout, target):
    if layout is None or target is None:
        return None
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item.widget() is target:
            return layout, index
        nested = item.layout()
        if nested is not None:
            found = _find_layout_with_widget(nested, target)
            if found is not None:
                return found
    return None
