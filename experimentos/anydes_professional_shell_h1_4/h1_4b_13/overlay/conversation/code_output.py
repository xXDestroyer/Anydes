"""Request-local structured code boundary; never writes source or executes it."""
from contextvars import ContextVar
from dataclasses import asdict
from functools import wraps
import hashlib
import json

_observer = ContextVar('code_generation_observer', default=None)
FIELDS = {'language': str, 'filename': str, 'code': str, 'tests': list}


class CodeOutputFailure(ValueError):
    def __init__(self, reason, **details):
        super().__init__(reason)
        self.reason = reason
        self.details = details


def record(owner, **values):
    # Diagnostics must not become a new failure path.
    try:
        owner.audit.record('code_generation_boundary', **values)
    except Exception:
        pass


def observe_clients(owner):
    router = getattr(getattr(getattr(owner, '_realtime_service', None), 'transport', None), 'router', None)
    for role, client in (('PRIMARY_GENERATION', getattr(router, 'online', None)),
                         ('FALLBACK_GENERATION', getattr(router, 'local', None))):
        if client is None or getattr(client, '_code_observed', False):
            continue
        original = client.stream

        def wrap(method, backend, target):
            @wraps(method)
            def observed(*args, **kwargs):
                callback = _observer.get()
                info = {'backend': backend, 'model': str(getattr(target, 'model',
                    getattr(getattr(target, 'transport', None), 'model', 'unknown')))}
                try:
                    outcome = method(*args, **kwargs)
                except Exception as exc:
                    if callback:
                        callback(dict(info, transport_success=False,
                            http_status=getattr(exc, 'status', None),
                            fallback_reason=str(getattr(exc, 'error_type', type(exc).__name__))))
                    raise
                if callback:
                    callback(dict(info, transport_success=not bool(getattr(outcome, 'error_type', '')),
                        http_status=None, fallback_reason=None))
                return outcome
            return observed
        client.stream = wrap(original, role, client)
        client._code_observed = True


def structure(text):
    """Count outer object spans without interpreting braces inside JSON strings."""
    depth = 0
    quoted = escaped = False
    count = 0
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"' and depth:
            quoted = True
        elif char == '{':
            if depth == 0:
                count += 1
            depth += 1
        elif char == '}' and depth:
            depth -= 1
    return {'outer_json_detected': count > 0, 'candidate_objects': count,
            'fenced_content': '```' in text, 'unclosed_object': depth > 0}


def decode(text, spec, extractor):
    from .code_generation import GeneratedCodeArtifact, CodeValidator
    shape = structure(text)
    if shape['candidate_objects'] > 1:
        raise CodeOutputFailure('CODE_GENERATION_AMBIGUOUS_STRUCTURED_OUTPUT')
    payload = extractor(text)
    missing = sorted(set(FIELDS) - set(payload))
    unexpected = sorted(set(payload) - set(FIELDS))
    invalid = sorted(k for k, kind in FIELDS.items() if k in payload and not isinstance(payload[k], kind))
    if missing or unexpected or invalid:
        raise CodeOutputFailure('CODE_GENERATION_SCHEMA_FAILURE', missing_fields=missing,
            unexpected_fields=unexpected, invalid_types=invalid)
    if payload['language'].casefold() != spec.language.casefold() or payload['filename'] != spec.target_filename:
        raise CodeOutputFailure('CODE_GENERATION_SEMANTIC_MISMATCH',
            invalid_fields=['language', 'filename'])
    payload['tests'] = [asdict(test) for test in spec.tests]
    artifact = GeneratedCodeArtifact.from_dict(payload)
    validation = CodeValidator.validate(artifact, spec)
    if not validation.valid:
        raise CodeOutputFailure('CODE_GENERATION_SCHEMA_FAILURE', invalid_fields=['code'],
            validation_reasons=list(validation.errors))
    return artifact


def generate(bridge, messages, spec, extractor):
    owner = bridge.assistant()
    observe_clients(owner)
    task_id = ''
    try:
        from .runtime_patch import _CONTROLLERS
        controller = _CONTROLLERS.get(owner)
        task_id = controller.state.pending_task.task_id if controller else ''
    except (AttributeError, TypeError):
        pass
    base = dict(task_id=task_id, operation='general_generate_code', capability='developer.write_code')
    current = messages
    for attempt in range(2):
        providers = []
        token = _observer.set(providers.append)
        try:
            _, outcome = bridge.complete(current, max_output_tokens=2600)
        except Exception as exc:
            record(owner, **base, attempt=attempt, providers=providers, transport_success=False,
                   exception_type=type(exc).__name__, repair_attempted=bool(attempt))
            raise
        finally:
            _observer.reset(token)
        text = outcome.text or ''
        finish = getattr(outcome, 'done_reason', None)
        evidence = dict(base, attempt=attempt, providers=providers,
            response_characters=len(text), response_bytes=len(text.encode('utf-8')),
            response_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
            stop_reason=finish, stop_reason_authority='transport; primary may normalize stop',
            repair_attempted=bool(attempt), **structure(text))
        if getattr(outcome, 'error_type', '') or getattr(outcome, 'timed_out', False) or getattr(outcome, 'connection_state', 'completed') != 'completed':
            record(owner, **evidence, transport_success=False, reason_code='CODE_GENERATION_TRANSPORT_FAILURE')
            raise CodeOutputFailure('CODE_GENERATION_TRANSPORT_FAILURE')
        if finish in {'length', 'max_tokens', 'max_output_tokens'}:
            record(owner, **evidence, transport_success=True, reason_code='CODE_GENERATION_OUTPUT_TRUNCATED')
            raise CodeOutputFailure('CODE_GENERATION_OUTPUT_TRUNCATED')
        try:
            artifact = decode(text, spec, extractor)
        except (ValueError, TypeError, SyntaxError) as exc:
            details = getattr(exc, 'details', {})
            reason = getattr(exc, 'reason', 'CODE_GENERATION_UNKNOWN_PARSE_FAILURE')
            if isinstance(exc, json.JSONDecodeError):
                reason = 'CODE_GENERATION_JSON_PARSE_FAILURE'
                details = dict(parser_exception=type(exc).__name__, parser_offset=exc.pos,
                    parser_line=exc.lineno, parser_column=exc.colno,
                    parse_family='INVALID_JSON_ESCAPING' if 'escape' in exc.msg.lower() else
                    ('MODEL_OUTPUT_NOT_JSON' if not evidence['outer_json_detected'] else 'UNKNOWN_JSON_PARSE_FAILURE'))
            record(owner, **evidence, transport_success=True, schema_status='INVALID',
                reason_code=reason, repair_result='FAILED' if attempt else 'NOT_ATTEMPTED', **details)
            if attempt:
                raise CodeOutputFailure('CODE_GENERATION_REPAIR_FAILED', underlying_reason=reason) from exc
            current = [messages[0], messages[1], {'role': 'user', 'content': json.dumps({
                'instruction': 'Reconstruct ONLY the required JSON object. Preserve original goal, language, '
                    'target_filename, all constraints and execution semantics. No extra fields. '
                    'Required string fields: language, filename, code. Required array: tests. '
                    'Escape source as a JSON string. Do not execute anything.',
                'error': reason, 'diagnostics': details, 'invalid_output': text}, ensure_ascii=False)}]
            continue
        record(owner, **evidence, transport_success=True, schema_status='VALID',
               repair_result='VALID' if attempt else 'NOT_ATTEMPTED')
        return artifact
