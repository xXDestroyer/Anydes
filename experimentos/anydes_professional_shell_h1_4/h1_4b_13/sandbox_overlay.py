"""Phase-local module selection; official runtime and permission checks stay unchanged."""
from pathlib import Path
import hashlib
import importlib.util
import sys

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'overlay/windows_sandbox.py'


def install_source(expected_hash=None):
    digest=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if expected_hash is not None and digest != expected_hash:
        raise RuntimeError('STAGING_B2_SOURCE_HASH_MISMATCH')
    import anydes_tools
    existing=sys.modules.get('anydes_tools.windows_sandbox')
    if existing is not None:
        if Path(existing.__file__).resolve()!=SOURCE.resolve():
            raise RuntimeError('STAGING_B2_REQUIRES_FRESH_PROCESS')
        return digest
    spec=importlib.util.spec_from_file_location('anydes_tools.windows_sandbox',SOURCE)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    anydes_tools.windows_sandbox=module
    return digest


def install():
    digest=install_source()
    from anydes_tools import executor
    original=executor._worker_environment
    if getattr(original,'_staging_b2_overlay',False):
        return
    def environment(*args,**kwargs):
        env=original(*args,**kwargs)
        env['PYTHONPATH']=str(HERE/'worker_bootstrap')+';'+env['PYTHONPATH']
        env['ANYDES_STAGING_B2_SOURCE_SHA256']=digest
        return env
    environment._staging_b2_overlay=True
    executor._worker_environment=environment
