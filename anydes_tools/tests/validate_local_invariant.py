"""One live diagnostic before fix, one focal UI reproduction after fix."""
import hashlib
import json
import os
import sys
from anydes_tools.tests.validate_valueerror_diagnostic import runner

BASE = runner.ROOT / 'anydes_tools/logs/local_invariant_phase2a'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/tests/test_local_natural_invariant.py',
    'anydes_tools/tests/validate_local_invariant.py']))


def headless_before():
    destination = BASE / 'before' / 'candidate'
    destination.mkdir(parents=True, exist_ok=True)
    os.environ['ANYDES_RUNTIME_KIND'] = 'CANDIDATE'
    os.environ['ANYDES_STAGING_DATA_ROOT'] = str(destination)
    os.environ['ANYDES_RUNTIME_WRITE_TRACE'] = '1'
    import probar_anydes_tools_conversacion_v001_5 as bootstrap
    from anydes_tools.tests.observe_candidate_g2 import install

    def run(mode, **unused):
        from runtime_session import runtime_session
        from anydes.interfaz.console import build_assistant
        from anydes.interfaz.turn_interaction import TurnToken
        from anydes.voice_input.core import UserInputEvent
        from anydes.voice_input.integration import InputAssistant
        from anydes_tools.conversation.runtime_patch import ensure_controller_for
        with runtime_session(mode):
            assistant = build_assistant()
            try:
                text = 'Hay algo que quisiera comentarte y necesito una mano.'
                with TurnToken('diagnostic-before').bind():
                    response = InputAssistant(assistant, UserInputEvent.keyboard(text)).respond_realtime(text)
                controller = ensure_controller_for(assistant, runner.ROOT)
                result = dict(surface='HEADLESS_NORMAL_BOOTSTRAP', model_responses_injected=False,
                    conversation_purpose=controller.last_resolution.conversation_purpose.value,
                    response_sha256=hashlib.sha256(response.encode()).hexdigest())
                (destination / 'result.json').write_text(json.dumps(result), encoding='utf-8')
            finally:
                assistant.close()
        return 0
    return bootstrap.main(headless_runner=run, before_layers=lambda: install(destination))


if __name__ == '__main__':
    if sys.argv[1] == 'headless-before':
        raise SystemExit(headless_before() or 0)
    runner.OUT = BASE / ('before' if sys.argv[1] == 'freeze-before' else 'ui')
    if sys.argv[1] == 'freeze-before':
        sys.argv[1] = 'freeze'
    raise SystemExit(runner.main() or 0)
