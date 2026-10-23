"""Deprecated Tk companion compatibility package.

The supported interface is ``anydes.interfaz.pyside_companion``. Imports stay
lazy so view-model compatibility does not load Tkinter.
"""

__all__ = ["CompanionApp", "main"]


def __getattr__(name: str):
    if name == "CompanionApp":
        from anydes.interfaz.companion.app import CompanionApp

        return CompanionApp
    if name == "main":
        return main
    raise AttributeError(name)


def main() -> None:
    from anydes.interfaz.pyside_companion.app import main as companion_main

    companion_main()
