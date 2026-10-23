"""Candidate-only module overlay. Never imported by the official launcher."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def install():
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(ROOT), str(ROOT / 'programa_principal')]
    import importlib.util
    import anydes_tools
    from sandbox_overlay import install as install_sandbox
    install_sandbox()
    overlay = str(HERE / 'overlay/conversation')
    if 'anydes_tools.conversation' in sys.modules:
        if overlay not in sys.modules['anydes_tools.conversation'].__path__:
            raise RuntimeError('H14B_OVERLAY_REQUIRES_FRESH_PROCESS')
        return
    original=ROOT/'anydes_tools/conversation'
    spec=importlib.util.spec_from_file_location('anydes_tools.conversation',original/'__init__.py',
        submodule_search_locations=[overlay,str(original)])
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    anydes_tools.conversation=module
    from anydes_tools.conversation.technical_literals import install as install_literals
    install_literals()


if __name__ == '__main__':
    install()
    import runpy
    runpy.run_path(str(HERE.parent / 'run_shell.py'), run_name='__main__')
