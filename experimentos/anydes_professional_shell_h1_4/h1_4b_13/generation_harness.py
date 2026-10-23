"""Exactly one live generation, no planner/tool dispatch or external execution."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import sys
import uuid
import bootstrap

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'evidence/real_generation'
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'programa_principal'),str(ROOT)]
if (OUT/'request_started.json').exists():
    raise SystemExit('Single real generation already attempted; do not retry')
OUT.mkdir(parents=True,exist_ok=True)
import anydes_hybrid_runtime as hybrid
from lanzar_anydes import load_profile,OFFICIAL_PROFILE,OFFICIAL_VISUAL_PROFILE
load_profile(OFFICIAL_PROFILE)
load_profile(OFFICIAL_VISUAL_PROFILE)
hybrid._load_non_secret_profile()
hybrid._load_secret()
if os.environ.get('ANYDES_HYBRID_FAULT_PLAN'):
    raise RuntimeError('No fault injection in live harness')
from anydes_tools.runtime_environment import RuntimeEnvironment,activate
activate(RuntimeEnvironment('STAGING',ROOT,OUT))
os.environ['ANYDES_HYBRID_LOG_DIR']=str(OUT/'hybrid')
os.environ['ANYDES_COGNITIVE_LOG_DIR']=str(OUT/'cognitive')
primary=os.getenv('ANYDES_PRIMARY_GROQ_MODEL','openai/gpt-oss-120b')
os.environ['PRIMARY_GROQ_MODEL']=primary
os.environ['ANYDES_REALTIME_MODEL']=f'groq:{primary}|fallback:qwen3:8b'
bootstrap.install()
hybrid._install_hybrid_transport()
from anydes.nucleo.config import Config
from anydes.realtime.transport import OllamaRealtimeTransport
from anydes.hybrid.router import close_all_routers
from anydes_tools.conversation.code_generation import CodeGenerationSpec,CodeValidator
from anydes_tools.conversation.semantic_runtime import AssistantCodeGenerator
from anydes_tools.conversation.runtime_patch import _CONTROLLERS

class Audit:
    def record(self,event,**values):
        with (OUT/'diagnostics.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=event,**values),ensure_ascii=True)+'\n')

class Owner:
    pass

config=Config()
owner=Owner()
owner.config=config
owner.audit=Audit()
owner.personality=object()
owner._realtime_service=SimpleNamespace(transport=OllamaRealtimeTransport(
    base_url=config.ollama_url,model=config.realtime_model,keep_alive=config.realtime_keep_alive,
    timeout_seconds=config.realtime_timeout_seconds,max_output_tokens=config.realtime_max_output_tokens,
    num_ctx=config.realtime_num_ctx))
task_id='task:'+uuid.uuid4().hex
_CONTROLLERS[owner]=SimpleNamespace(state=SimpleNamespace(pending_task=SimpleNamespace(task_id=task_id)))
spec=CodeGenerationSpec.from_dict(dict(language='python',goal='Create a console letter counter',
    functional_requirements=['Read one sentence from standard input',
        'Count alphabetic characters only, excluding spaces, digits, punctuation',
        'Print the total number of letters'],inputs=['one sentence'],outputs=['letter count'],
    constraints=['No external execution requested','Use standard Python without dependencies'],
    dependencies=[],target_filename='letter_counter.py',tests=[],execution_required=False,
    verification_requirement='BASIC',user_interactive=True))
result=dict(passed=False,task_id=task_id,ui_opened=False,external_execution=False,
    tool_calls=0,artifact_committed=False,source_proposal_produced=False)
try:
    (OUT/'request_started.json').write_text(json.dumps({'structured_requests_maximum':2,
        'dedicated_repairs_maximum':1,'task_id':task_id}),encoding='utf-8')
    artifact=AssistantCodeGenerator(owner)(spec)
    validation=CodeValidator.validate(artifact,spec)
    result.update(passed=validation.valid,source_proposal_produced=True,
        source_sha256=hashlib.sha256(artifact.code.encode()).hexdigest(),
        filename=artifact.filename,language=artifact.language,validation_errors=list(validation.errors))
    if validation.valid:
        (OUT/artifact.filename).write_text(artifact.code,encoding='utf-8')
except Exception as exc:
    result.update(exception_type=type(exc).__name__,reason_code=getattr(exc,'reason',type(exc).__name__),
        details=getattr(exc,'details',{}))
finally:
    close_all_routers()
    _CONTROLLERS.pop(owner,None)
    (OUT/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
raise SystemExit(0 if result['passed'] else 1)
