"""Candidate-only exception observation; never changes outcomes or retries."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
from pathlib import Path
import uuid


_active = ContextVar('conversation_diagnostics', default=None)
SAFE_MESSAGES = frozenset({
    'conversational_reply_schema_invalid', 'natural_response_local_invariant_rejected',
    'natural_response_grounding_rejected', 'subjective_reply_offer_not_allowed',
    'review_fields_invalid', 'review_values_invalid', 'review_semantics_invalid',
    'CONVERSATIONAL_REVIEW_UNAVAILABLE', 'REVIEWER_PROVIDER_UNAVAILABLE',
    'REVIEWER_RESPONSE_INCOMPLETE', 'NO_CONFIGURED_REVIEWER_FALLBACK',
})
VALIDATION_CODES = frozenset({
    'user_echo_removed', 'question_echo_removed', 'resolved_question_repetition_removed',
    'unsupported_self_history_removed', 'generic_tail_removed', 'generic_response_only',
    'empty_output', 'technical_leak', 'internal_label', 'unverified_action_claim',
    'unbacked_task_commitment', 'unverified_visual_claim', 'question_treated_as_memory',
    'unsupported_physical_experience', 'user_self_preference_mixing',
    'self_preference_unverified', 'self_preference_contradiction',
    'mirrors_user_preference_as_self', 'mirrors_user_state_as_self',
    'user_preference_unverified', 'length_trimmed',
})


class ConversationDiagnostics:
    def __init__(self, record, root, turn_id=''):
        self.record, self.root = record, Path(root).resolve()
        self.turn_id = str(turn_id)
        self.correlation_id = uuid.uuid4().hex

    def exception(self, stage, error):
        chain, seen = [], set()
        current = error
        while current is not None and id(current) not in seen and len(chain) < 4:
            seen.add(id(current))
            frames, tb = [], current.__traceback__
            while tb is not None:
                filename = Path(tb.tb_frame.f_code.co_filename)
                try:
                    file = filename.resolve().relative_to(self.root).as_posix()
                except ValueError:
                    file = '$EXTERNAL/' + filename.name
                frames.append(dict(file=file, function=tb.tb_frame.f_code.co_name,
                                   line=tb.tb_lineno))
                tb = tb.tb_next
            message = str(current)
            # Arbitrary exception messages may contain prompts, keys or personal
            # values. Retain only known static codes and standard JSON diagnostics.
            safe = message if message in SAFE_MESSAGES else '[REDACTED_EXCEPTION_TEXT]'
            item = dict(exception_type=type(current).__name__, sanitized_message=safe,
                        message_sha256=hashlib.sha256(message.encode()).hexdigest(),
                        traceback=frames, callsite=frames[-1] if frames else None)
            if isinstance(current, json.JSONDecodeError):
                item.update(sanitized_message='JSON_DECODE_ERROR',
                            json_line=current.lineno, json_column=current.colno,
                            json_position=current.pos)
            chain.append(item)
            current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
        try:
            self.record('conversation_boundary_exception', stage=stage,
                        turn_id=self.turn_id, correlation_id=self.correlation_id,
                        **chain[0], exception_chain=chain[1:])
        except Exception:
            # Diagnostic I/O must not replace the original exception or outcome.
            import logging
            logging.getLogger(__name__).error('conversation_diagnostic_write_failed')

    @contextmanager
    def stage(self, name):
        token = _active.set(self)
        try:
            yield
        except Exception as error:
            self.exception(name, error)
            raise
        finally:
            _active.reset(token)

    def guard(self, name, function):
        @wraps(function)
        def observed(*args, **kwargs):
            with self.stage(name):
                return function(*args, **kwargs)
        return observed


def current_or_new(assistant, root, turn_id=''):
    return _active.get() or ConversationDiagnostics(assistant.audit.record, root, turn_id)


@contextmanager
def boundary(name):
    diagnostic = _active.get()
    if diagnostic is None:
        yield
    else:
        with diagnostic.stage(name):
            yield


def observed_entry(function, root, turn_id):
    @wraps(function)
    def observed(assistant, *args, **kwargs):
        diagnostic = current_or_new(assistant, root, turn_id())
        with diagnostic.stage('REALTIME_UI_BRIDGE'):
            return function(assistant, *args, **kwargs)
    return observed


def record_local_rejection(text, validation):
    """Only structural flags and known validator codes, never rejected prose."""
    diagnostic = _active.get()
    if diagnostic is None:
        return
    issues = tuple(getattr(validation, 'issues', ()))
    try:
        diagnostic.record('conversation_local_gate_rejected',
            stage='LOCAL_REVIEW_GATE', turn_id=diagnostic.turn_id,
            correlation_id=diagnostic.correlation_id,
            candidate_type=type(text).__name__,
            candidate_empty=not isinstance(text, str) or not text.strip(),
            candidate_protocol_prefix=isinstance(text, str) and text.lstrip().startswith(('{', '```json')),
            validation_present=validation is not None,
            validation_valid=getattr(validation, 'valid', None),
            validation_codes=[code for code in issues if code in VALIDATION_CODES],
            other_validation_codes=sum(code not in VALIDATION_CODES for code in issues))
    except Exception:
        import logging
        logging.getLogger(__name__).error('conversation_diagnostic_write_failed')
