"""Metadata-only staging observation; never selects or authorizes an operation."""
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import hashlib
import re

STAGES = ('INTERACTION_RESOLUTION', 'SEMANTIC_PLANNING', 'PLAN_LOWERING',
          'PLAN_VALIDATION', 'PERMISSION_CHECK', 'PENDING_CONFIRMATION',
          'EXECUTION_DISPATCH', 'B2_EXECUTION', 'TESTSPEC_BUILD',
          'VERIFICATION_DISPATCH', 'ORACLE_SELECTION', 'COMMIT', 'PUBLICATION')


def exception_code(exc):
    value = str(exc)
    # Known local contract errors only. Arbitrary exception text stays private.
    prefixes = ('test_spec_', 'test_', 'operation_descriptor_', 'operation_semantics_',
                'verification_', 'PLAN_', 'DATAFLOW_', 'CAPABILITY_', 'UNKNOWN_CAPABILITY')
    if value.startswith(prefixes) and re.fullmatch(r'[A-Za-z0-9_:.;-]{1,240}', value):
        return value
    return 'UNEXPECTED_EXCEPTION_' + type(exc).__name__


def exception_metadata(exc):
    frames, tb = [], exc.__traceback__
    while tb is not None:
        frames.append({'module': tb.tb_frame.f_globals.get('__name__', ''),
                       'function': tb.tb_frame.f_code.co_name, 'line': tb.tb_lineno})
        tb = tb.tb_next
    return {'exception_type': type(exc).__name__, 'frames': frames[-8:],
            'message_sha256': hashlib.sha256(str(exc).encode()).hexdigest()}


class PipelineTrace:
    def __init__(self, audit, turn_id, identity):
        self.audit, self.turn_id, self.identity = audit, turn_id, identity
        self.seen, self.failed_stage, self.failed_reason = set(), '', ''
        self.current = []
        self.last_exited = ''

    def record(self, stage, *, entered=False, exited=False, status='', reason_code='', next_stage=''):
        self.seen.add(stage)
        task_id, revision = self.identity()
        self.audit.record('pipeline_stage_trace', task_id=task_id, plan_revision=revision,
                          turn_id=self.turn_id, stage=stage, entered=entered, exited=exited,
                          status=status, reason_code=reason_code, next_stage=next_stage,
                          timestamp=datetime.now(timezone.utc).isoformat())

    @contextmanager
    def stage(self, name):
        if self.last_exited and not self.current:
            self.record(self.last_exited, status='TRANSITION', next_stage=name)
        self.record(name, entered=True, status='ENTERED')
        self.current.append(name)
        try:
            yield
        except Exception as exc:
            if not self.failed_stage:
                self.failed_stage, self.failed_reason = name, exception_code(exc)
            self.record(name, exited=True, status='FAILED', reason_code=exception_code(exc))
            raise
        else:
            self.record(name, exited=True, status='EXITED')
        finally:
            self.current.pop()
            self.last_exited = name

    def invoke(self, name, function, *args, **kwargs):
        with self.stage(name):
            result = function(*args, **kwargs)
            if getattr(result, 'valid', True) is False:
                self.record(name, status='FAILED', reason_code='PLAN_VALIDATION_REJECTED')
            return result

    def terminal(self, outcome):
        for name in STAGES[:-1]:
            if name not in self.seen:
                self.record(name, status='NOT_REACHED', reason_code='TERMINAL_BEFORE_STAGE')
        self.audit.record('pipeline_terminal', turn_id=self.turn_id, outcome_id=outcome.outcome_id,
                          task_id=outcome.task_id, plan_revision=outcome.plan_revision,
                          status=outcome.status.value, failure_ref=outcome.failure_ref,
                          reason_code=outcome.safe_evidence.get('reason_code', ''),
                          stage=outcome.safe_evidence.get('failure_stage') or
                          ('PENDING_CONFIRMATION' if outcome.status.value == 'PENDING_CONFIRMATION' else self.last_exited))


def traced_stage(name):
    def decorate(function):
        @wraps(function)
        def call(self, *args, **kwargs):
            trace = getattr(self, '_pipeline_trace', None)
            if trace is None:
                return function(self, *args, **kwargs)
            return trace.invoke(name, function, self, *args, **kwargs)
        return call
    return decorate
