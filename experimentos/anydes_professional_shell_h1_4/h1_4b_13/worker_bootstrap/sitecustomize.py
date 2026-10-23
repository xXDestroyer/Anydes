"""Only selected by the isolated staging worker environment."""
from pathlib import Path
import os
import sys

expected=os.environ.get('ANYDES_STAGING_B2_SOURCE_SHA256')
if expected:
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    try:
        from sandbox_overlay import install_source
        install_source(expected)
    except Exception:
        # Python otherwise ignores sitecustomize exceptions and could run the wrong module.
        raise SystemExit('STAGING_B2_OVERLAY_UNAVAILABLE')
