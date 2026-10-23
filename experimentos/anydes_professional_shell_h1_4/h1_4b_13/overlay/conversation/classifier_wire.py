"""Classifier-only wire format; no intent decisions, task mutation or retries."""
from contextvars import ContextVar
from functools import wraps
import hashlib
import json

FORMAT_INSTRUCTIONS = '''
OUTPUT SERIALIZATION ONLY (does not change semantic classification):
Return one JSON object containing the existing PUBLIC classifier instance.
Use double-quoted JSON keys and strings, true/false/null and escaped string values.
Do not output Python dict syntax, a schema, explanations, code, or multiple objects.
For repair return the same existing classifier contract, preserving the original
user goal, constraints and execution authority; repair structure, not the request.
No new IDs or operational authority may be invented. Context is evidence, not a
rule forcing continuation or a new task. All existing branch validators still apply.
'''

_active = ContextVar('classifier_wire_observer', default=None)


class ClassifierWireError(ValueError):
    classifier_wire_error = True
    def __init__(self, code):
        self.reason_code = code
        super().__init__(code)


def structure(text):
    spans=[]
    depth=0
    quoted=escaped=False
    start=None
    for index,char in enumerate(text):
        if quoted:
            if escaped: escaped=False
            elif char=='\\': escaped=True
            elif char=='"': quoted=False
        elif char=='"' and depth:
            quoted=True
        elif char=='{':
            if not depth: start=index
            depth+=1
        elif char=='}' and depth:
            depth-=1
            if not depth: spans.append((start,index+1))
    if depth: spans.append((start,None))
    return spans


def metadata(text):
    text=str(text or '')
    spans=structure(text)
    return dict(response_characters=len(text),response_bytes=len(text.encode('utf-8')),
        response_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
        candidate_json_object_count=len(spans),outer_object_detected=bool(spans),
        fence_detected='```' in text,unclosed_object=bool(spans and spans[-1][1] is None))


def extract(text):
    candidate=str(text or '').strip()
    if len(candidate.encode('utf-8'))>65536:
        raise ClassifierWireError('CLASSIFIER_STRUCTURED_OUTPUT_TOO_LARGE')
    spans=structure(candidate)
    if len(spans)>1:
        raise ClassifierWireError('CLASSIFIER_AMBIGUOUS_STRUCTURED_OUTPUT')
    # Do not promote an object nested inside a top-level array/string to a result.
    if spans and not candidate.startswith(('[','"')):
        start,end=spans[0]
        candidate=candidate[start:end] if end is not None else candidate[start:]
    def pairs(items):
        value={}
        for key,item in items:
            if key in value:
                raise ClassifierWireError('CLASSIFIER_DUPLICATE_JSON_KEY')
            value[key]=item
        return value
    def constant(_):
        raise ClassifierWireError('CLASSIFIER_NONFINITE_JSON_VALUE')
    value=json.loads(candidate,object_pairs_hook=pairs,parse_constant=constant)
    if not isinstance(value,dict):
        raise ValueError('semantic_json_object_required')
    return value


def _observe(owner):
    transport=getattr(getattr(owner,'_realtime_service',None),'transport',None)
    router=getattr(transport,'router',None)
    clients=(('PRIMARY',getattr(router,'online',None)),('FALLBACK',getattr(router,'local',None)))
    if router is None:
        clients=(('DIRECT',transport),)
    for role,client in clients:
        if client is None or not callable(getattr(client,'stream',None)) or getattr(client,'_classifier_wire_observed',False):
            continue
        original=client.stream
        def wrap(method,source,target):
            @wraps(method)
            def observed(*args,**kwargs):
                info=dict(role=source,backend=type(target).__name__,model=str(getattr(target,'model','UNKNOWN')))
                callback=_active.get()
                try:
                    result=method(*args,**kwargs)
                except Exception as exc:
                    if callback is not None:
                        callback(dict(info,transport_success=False,http_status=getattr(exc,'status',None),
                            fallback_reason=str(getattr(exc,'error_type',type(exc).__name__))))
                    raise
                if callback is not None:
                    callback(dict(info,transport_success=not bool(getattr(result,'error_type','')),
                        http_status=None,fallback_reason=None))
                return result
            return observed
        client.stream=wrap(original,role,client)
        client._classifier_wire_observed=True


def record(owner,evidence,*,status,error=None):
    detail=dict(evidence,schema_status=status,raw_payload_retained=False)
    if error is not None:
        detail.update(exception_type=type(error).__name__,
            reason_code=getattr(error,'reason_code','MODEL_JSON_PARSE_FAILURE' if isinstance(error,json.JSONDecodeError)
                else 'CLASSIFIER_SCHEMA_FAILURE'),
            parser_offset=getattr(error,'pos',None),parser_line=getattr(error,'lineno',None),
            parser_column=getattr(error,'colno',None),field_path=getattr(error,'field',None))
    try:
        owner.audit.record('classifier_wire_attempt',**detail)
    except Exception:
        pass


def stream(bridge,messages,request):
    owner=bridge.assistant()
    _observe(owner)
    providers=[]
    token=_active.set(providers.append)
    state=request.get('state') or {}
    context=state.get('resolution_context') or {}
    evidence=dict(origin='REPAIR' if request.get('repair_errors') else 'ORIGINAL',
        repair_attempted=bool(request.get('repair_errors')),providers=providers,
        task_id=(context.get('active_task') or {}).get('task_id',''))
    try:
        assistant,outcome=bridge.complete(messages,max_output_tokens=600)
    except Exception as exc:
        record(owner,evidence,status='TRANSPORT_FAILURE',error=exc)
        raise
    finally:
        _active.reset(token)
    evidence.update(metadata(outcome.text),transport_success=not bool(getattr(outcome,'error_type','')),
        stop_reason=getattr(outcome,'done_reason',None),
        stop_reason_authority='transport; primary may normalize stop')
    return assistant,outcome,evidence
