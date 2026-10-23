"""One real classifier continuation over a prepared, validated incomplete task."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys
import bootstrap

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'evidence/continuation_harness'
if (OUT/'request_started.json').exists():
    raise SystemExit('Single controlled continuation already attempted')
OUT.mkdir(parents=True,exist_ok=True)
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'programa_principal'),str(ROOT)]
import anydes_hybrid_runtime as hybrid
from lanzar_anydes import load_profile,OFFICIAL_PROFILE,OFFICIAL_VISUAL_PROFILE
load_profile(OFFICIAL_PROFILE)
load_profile(OFFICIAL_VISUAL_PROFILE)
hybrid._load_non_secret_profile()
hybrid._load_secret()
if os.environ.get('ANYDES_HYBRID_FAULT_PLAN'):
    raise RuntimeError('No failure injection in the live classifier harness')
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
from anydes_tools.conversation.semantic_runtime import AssistantInteractionClassifier
from anydes_tools.conversation.interaction_router import InteractionResolver,InteractionResolution
from anydes_tools.conversation.operation_intent import semantic_resolution_payload
from anydes_tools.conversation.task_continuity import TaskContract

empty={'pending_action':False,'failure_available':False,'retry_available':False,
    'resolution_context':{'active_task':{},'pending_confirmation':{'available':False}}}
questions={'language':'Which programming language?', 'filename':'What source filename?',
    'input_mode':'How is the sentence supplied?', 'counting_policy':'Which characters count, and how is case handled?'}
proposal={'relation':'START','operation':'CREATE_NEW','artifact_type':'CODE','requirements':[],
    'requirement_items':[{'key':key,'necessity':'REQUIRED','question':question} for key,question in questions.items()],
    'planning_readiness':{'goal_resolvable':True,'artifact_specified':True,'material_contradictions':[]},
    'clarification_target':{'key':'language','question':questions['language']}}
initial={'intent':'CLARIFICATION','speech_act':'INFORMATION_REQUEST',
    'primary_goal':'Create a program that counts letters in a sentence','confidence':.98,
    'artifact_type':'CODE','execute':False,'verification_requirement':'BASIC',
    'clarification_scope':'TASK','task_proposal':proposal,'turn_relation':'NEW_TOPIC'}
decision=InteractionResolution.from_dict(semantic_resolution_payload(initial,empty))
task=TaskContract.start(decision,decision.task_proposal)
task.merge(decision.task_proposal,'controlled:turn1')
state={'pending_action':False,'failure_available':False,'retry_available':False,
    'pending_status':'AWAITING_CLARIFICATION','active_artifact':False,'recent_artifacts':[],
    'resolution_context':{'active_task':{'task_id':task.task_id,'status':'AWAITING_CLARIFICATION',
        'semantic_contract':task.snapshot()},'previous_assistant_act':'CLARIFICATION_REQUEST',
        'active_failure_topic':{},'pending_confirmation':{'available':False},
        'discourse':{'purpose':'GENERAL','subjects':[]},
        'recent_dialogue':[{'role':'user','content':'Creame un programa que cuente letras de una oracion.'},
            {'role':'assistant','content':'What language, filename, input and counting rules should this program use?'}]}}

class Audit:
    def record(self,event,**values):
        with (OUT/'diagnostics.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=event,**values),ensure_ascii=True)+'\n')
class Owner:
    pass
config=Config()
owner=Owner()
owner.config=config
owner.personality=object()
owner.audit=Audit()
owner._realtime_service=SimpleNamespace(transport=OllamaRealtimeTransport(
    base_url=config.ollama_url,model=config.realtime_model,keep_alive=config.realtime_keep_alive,
    timeout_seconds=config.realtime_timeout_seconds,max_output_tokens=config.realtime_max_output_tokens,
    num_ctx=config.realtime_num_ctx))
resolver=InteractionResolver(AssistantInteractionClassifier(owner))
result=dict(passed=False,prepared_turn1=True,live_classifier_turns=1,task_id=task.task_id,
    tool_calls=0,b2=False,ui_opened=False,artifact_creation=False)
try:
    (OUT/'request_started.json').write_text(json.dumps({'maximum_attempts':2,'maximum_repairs':1,
        'task_id':task.task_id}),encoding='utf-8')
    followup=resolver.resolve('Usa Python y el nombre letter_counter.py. La oracion se introduce por consola. '
        'Cuenta solo letras de la a a la z, incluidas mayusculas; ignora espacios, numeros y puntuacion. '
        'No lo ejecutes como accion para mi.',state=state)
    update=followup.task_update
    result.update(source=followup.source,semantic_intent=followup.semantic_intent,
        task_update_present=bool(update),task_id_matches=bool(update and update.get('task_id')==task.task_id),
        execution_requested=followup.requires_execution,attempts=resolver.validation_history)
    valid=(followup.semantic_intent=='CONTINUE_TASK' and result['task_id_matches']
        and not followup.requires_execution and followup.task_proposal is None)
    if valid:
        task.merge(update,'controlled:turn2')
        result.update(same_task_after_merge=task.task_id==result['task_id'],
            requirement_keys=sorted(task.requirements),unresolved=list(task.unresolved_questions))
    result['passed']=bool(valid and result.get('same_task_after_merge'))
except Exception as exc:
    result.update(exception_type=type(exc).__name__)
finally:
    close_all_routers()
    (OUT/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
raise SystemExit(0 if result['passed'] else 1)
