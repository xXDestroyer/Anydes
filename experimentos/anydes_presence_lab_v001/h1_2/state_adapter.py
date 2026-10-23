"""Read-only projection of typed runtime activity. No core imports or commands."""
from dataclasses import dataclass
from enum import Enum
import hashlib
import re
import time
from datetime import datetime, timezone


class State(str, Enum):
    IDLE = 'IDLE'
    LISTENING = 'LISTENING'
    THINKING = 'THINKING'
    SPEAKING = 'SPEAKING'
    WORKING = 'WORKING'


class Domain(str, Enum):
    CAPTURE = 'CAPTURE'
    STT = 'STT'
    INFERENCE = 'INFERENCE'
    SYNTHESIS = 'SYNTHESIS'
    PLAYBACK = 'PLAYBACK'
    EXECUTION = 'EXECUTION'
    VERIFY = 'VERIFY'
    COMMIT = 'COMMIT'


MAPPING = {
    Domain.CAPTURE: (State.LISTENING, 'voice.capture'),
    Domain.STT: (State.LISTENING, 'voice.stt'),
    Domain.INFERENCE: (State.THINKING, 'brain.inference'),
    Domain.SYNTHESIS: (State.THINKING, 'voice.synthesis'),
    Domain.PLAYBACK: (State.SPEAKING, 'b4.playback'),
    Domain.EXECUTION: (State.WORKING, 'tools.execution'),
    Domain.VERIFY: (State.WORKING, 'tools.verification'),
    Domain.COMMIT: (State.WORKING, 'tools.commit'),
}
PRIORITY = {State.IDLE: 0, State.THINKING: 1, State.WORKING: 2, State.LISTENING: 3, State.SPEAKING: 4}


def opaque(value):
    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()[:16]


@dataclass(frozen=True)
class RuntimeEvent:
    domain: Domain
    phase: str
    activity_id: str
    task_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.domain, Domain) or self.phase not in {'START', 'END'}:
            raise ValueError('invalid_activity_contract')
        if not re.fullmatch('[a-f0-9]{16}', self.activity_id):
            raise ValueError('activity_id_must_be_opaque')
        if self.task_id is not None and not re.fullmatch('[a-f0-9]{16}', self.task_id):
            raise ValueError('task_id_must_be_opaque')


class PresenceStateAdapter:
    def __init__(self, *, clock=time.monotonic, settle_seconds=0.12, session=None):
        if not 0 <= settle_seconds <= 0.2:
            raise ValueError('coalescing_budget_exceeded')
        self.clock, self.settle = clock, settle_seconds
        self.session = session or opaque(time.time_ns())
        self.active = {}
        self.connected = False
        self.sequence = 0
        self.state, self.source, self.task_id = State.IDLE, 'runtime.disconnected', None
        self.idle_due = None
        self.last_transition = self.clock()

    def ready(self):
        self.connected = True
        return self._resolve(force=True)

    def disconnect(self):
        self.connected = False
        self.active.clear()
        self.idle_due = None
        return self._publish(State.IDLE, 'runtime.disconnected', None, force=True)

    def apply(self, event):
        if not self.connected:
            return None
        key = (event.domain, event.activity_id)
        if event.phase == 'START':
            self.active[key] = event
        else:
            self.active.pop(key, None)
        return self._resolve()

    def _resolve(self, force=False):
        if self.active:
            event = max(self.active.values(), key=lambda e: PRIORITY[MAPPING[e.domain][0]])
            self.idle_due = None
            state, source = MAPPING[event.domain]
            return self._publish(state, source, event.task_id, force=force)
        if self.state != State.IDLE and not force:
            if self.idle_due is None:
                self.idle_due = self.clock() + self.settle
            return None
        return self._publish(State.IDLE, 'runtime.idle', None, force=force)

    def advance(self):
        if self.idle_due is not None and self.clock() >= self.idle_due and not self.active:
            self.idle_due = None
            return self._publish(State.IDLE, 'runtime.idle', None)
        return None

    def _publish(self, state, source, task_id, force=False):
        if not force and (state, source, task_id) == (self.state, self.source, self.task_id):
            return None
        self.state, self.source, self.task_id = state, source, task_id
        self.sequence += 1
        self.last_transition = self.clock()
        return self.snapshot()

    def snapshot(self):
        return dict(schema='anydes.presence.v1', session=self.session, sequence=self.sequence,
                    state=self.state.value, source=self.source, task_id=self.task_id,
                    timestamp=datetime.now(timezone.utc).isoformat(), connected=self.connected)
