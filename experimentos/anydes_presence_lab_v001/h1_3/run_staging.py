"""H1.3 uses the accepted isolated host with a different observation provider."""
from pathlib import Path
import sys
from datetime import datetime, timezone
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0,str(HERE.parent/'h1_2'))
import run_staging as host
import event_service
import runtime_observer
from runtime_motion import VoiceMotionService, MotionObserver
host.SESSION = HERE/'sessions'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
event_service.PresenceService = VoiceMotionService
runtime_observer.RuntimeObserver = MotionObserver
if __name__ == '__main__':
    raise SystemExit(host.main())
