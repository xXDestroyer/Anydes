"""Isolated observer diagnostic; no assistant, models, audio or human UI replay."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'programa_principal')]
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def main():
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication, QWidget
    from unittest.mock import patch
    import anydes_tools.conversation.controller  # Real parity hash targets.
    from anydes_tools.tests.observe_candidate_g2 import install

    output = ROOT / 'anydes_tools/logs/operation_grounding/lifecycle_diagnostic'
    output.mkdir(exist_ok=True)
    temporary = output / '.fixture_state.json.atomic.tmp'
    target = output / 'fixture_state.json'
    temporary.write_text('{}', encoding='utf-8')
    application = QApplication([])
    window = QWidget()
    window.show()  # Offscreen fixture, not Anydes UI.
    install(output)
    original_read = Path.read_bytes
    original_exec = QApplication.exec
    renamed = False

    def read_after_atomic_rename(path):
        nonlocal renamed
        if not renamed and path.name == 'interaction_router.py':
            os.replace(temporary, target)
            renamed = True
        return original_read(path)

    result = {'kind': 'ISOLATED_CONTROLLED_DIAGNOSTIC_NOT_HUMAN_REPLAY',
              'injected_condition': 'atomic rename after enumeration before stat',
              'candidate_changed': False}
    try:
        with patch.object(Path, 'read_bytes', read_after_atomic_rename):
            application.exec()
    except FileNotFoundError as error:
        result.update(exception=type(error).__name__, missing_file=Path(error.filename).name,
                      initial_snapshot_escaped=True)
        # A modal dialog can pump the observer's timer even though the main
        # QApplication.exec original was never reached. No audio worker here.
        loop = QEventLoop()
        QTimer.singleShot(5300, loop.quit)
        loop.exec()
    finally:
        window.close()
    events = [json.loads(line) for line in (output / 'lifecycle.jsonl').read_text().splitlines()]
    result['renamed'] = renamed
    result['event_loop_start_recorded'] = any(r.get('phase') == 'EVENT_LOOP_START' for r in events)
    result['event_loop_return_recorded'] = any(r.get('phase') == 'EVENT_LOOP_RETURN' for r in events)
    result['periodic_snapshot_in_nested_loop'] = any(r.get('phase') == 'PERIODIC' for r in events)
    result['observer_wrapper_unchanged'] = QApplication.exec is original_exec
    (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    assert result.get('initial_snapshot_escaped') and renamed
    assert not result['event_loop_start_recorded'] and not result['event_loop_return_recorded']
    assert result['periodic_snapshot_in_nested_loop']


if __name__ == '__main__':
    main()
