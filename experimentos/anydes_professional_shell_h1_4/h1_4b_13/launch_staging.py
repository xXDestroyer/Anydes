"""Visible staging only; reuse the accepted startup observer without mutation."""
from pathlib import Path
import runpy
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=BASE.parents[1]
sys.dont_write_bytecode=True
sys.path[:0]=[str(HERE),str(BASE),str(ROOT/'programa_principal'),str(ROOT),str(BASE/'h1_4b_8')]
if Path(sys.prefix).resolve()!=(ROOT/'.venv').resolve():
    raise SystemExit('STAGING_EXPECTS_PROJECT_VENV')
import bootstrap
bootstrap.install()
from startup_observer import install,record
record('launcher_started',candidate='h1_4b_13')
install()
try:
    runpy.run_path(str(BASE/'run_shell.py'),run_name='__main__')
except SystemExit as error:
    record('launcher_exit',exit_code=error.code if isinstance(error.code,int) else None)
    raise
except BaseException as error:
    record('launcher_exception',exception_type=type(error).__name__)
    raise
