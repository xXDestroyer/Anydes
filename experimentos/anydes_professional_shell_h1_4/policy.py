"""UI-only selection and read-only presence projection."""
from enum import Enum
import json
from pathlib import Path


class RendererKind(str, Enum):
    HOLOGRAM = 'HOLOGRAM'
    AVATAR_2_5D = 'AVATAR_2_5D'


STATE_LABELS = dict(IDLE='En reposo', LISTENING='Escuchando...', THINKING='Pensando...',
                    SPEAKING='Hablando...', WORKING='Trabajando...')
WORK_LABELS = dict(PROGRAMMING='Programando...', CREATING_FILE='Creando archivo...',
                   EDITING='Editando...', EXECUTING='Ejecutando...',
                   VERIFYING='Verificando...', CONVERTING='Convirtiendo...')


def status_label(state, mode='NONE', connected=True):
    if not connected:
        return 'Presencia en pausa'
    return WORK_LABELS.get(mode, STATE_LABELS['WORKING']) if state == 'WORKING' else STATE_LABELS.get(state, 'En reposo')


def load_selection(sessions):
    # Preferences live only in staging sessions, never in SELF or Memory.
    for path in sorted(Path(sessions).glob('*/ui_selection.json'), reverse=True):
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            return RendererKind(value['renderer']) if value.get('remember') else None
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def save_selection(session, kind, remember):
    path = Path(session) / 'ui_selection.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(schema=1, renderer=RendererKind(kind).value,
                                         remember=bool(remember))), encoding='utf-8')
    temporary.replace(path)


def can_switch(state, connected, worker_running=False):
    return connected and state == 'IDLE' and not worker_running
