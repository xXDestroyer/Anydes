"""One model-facing semantic contract, projected locally into runtime operations."""
PROMPT = '''Resuelve SOLO la intencion del turno actual en su contexto. Devuelve JSON estricto.
Campos obligatorios: intent, speech_act, primary_goal, confidence (0..1 o LOW/MEDIUM/HIGH).
Para CONVERSATION y PRE_REQUEST declara tambien conversation_purpose: GENERAL o PRE_REQUEST.
Es el proposito discursivo, no una operacion. Debe coincidir con el intent declarado.

ANTES de elegir CONVERSATION, distingue la funcion comunicativa del turno completo:
- PRE_REQUEST: el usuario abre espacio para una consulta/peticion futura y espera que le
  cedas la palabra. Aun no ha encargado un objetivo concreto. Incluye anuncios de ayuda,
  introducciones a una consulta o peticiones de permiso conversacional para continuar.
  No depende de vocabulario ni de que la oracion tenga forma gramatical de pregunta.
  Usa intent=PRE_REQUEST, conversation_purpose=PRE_REQUEST, speech_act=SOCIAL o ACKNOWLEDGEMENT.
  primary_goal describe invitar a continuar; no inventes el asunto que aun no ha especificado.
- GENERAL: el turno YA aporta charla casual, una observacion social o un tema para conversar;
  no esta reservando un turno posterior para especificar una peticion.
  Usa intent=CONVERSATION, conversation_purpose=GENERAL y el speech_act adecuado.
- CLARIFICATION: el usuario YA intenta encargar algo concreto, pero falta un referente,
  parametro u objetivo necesario. Identifica lo entendido en primary_goal y lo ausente en
  ambiguities. No convertir una apertura de conversacion en una tarea incompleta.
Usa previous_assistant_act, discourse/current topic, recent_dialogue y turn_relation como
evidencia; no heredes una accion, fallo o aprobacion si el turno no se refiere a ellos.
Una accion concreta, aceptacion pendiente o pregunta causal sobre un fallo conserva su
dominio propio: no es PRE_REQUEST por tener tono cortés o pedir ayuda.
PRE_REQUEST no tiene efectos, referencias operativas, ejecucion ni verificacion.
El host aporta ACKNOWLEDGE_AND_WAIT a la respuesta natural; tu SOLO clasificas.

Campos opcionales: artifact_type (CODE/DOCUMENT/TEXT/ARCHIVE), references (array de nombres o referencias),
execute (boolean), verification_requirement (NONE/BASIC/EXECUTION/TEST_SUITE/STRUCTURAL/CONVERSION_PRESERVATION),
execution_intent (NONE/RUN/TEST),
turn_relation, subjects (array {name,semantic_type}), ambiguities (array de strings), explanation_relation.
identity_attribute es obligatorio SOLO para PERSONAL_IDENTITY: DISPLAY_NAME o UNSUPPORTED.
pending_action_relation solo al evaluar un turno respecto a pending_confirmation:
ACCEPT_PENDING_ACTION, REJECT_PENDING_ACTION, NEW_ACTION, AMBIGUOUS o UNRELATED.
NO emitas kind, task_control, requested_operations, output_kind ni requested_effects: son derivados locales.
El contrato es discriminado por intent + speech_act. Usa solo campos aplicables al dominio.
ACTION_REQUEST operacional: omite explanation_relation; no es parte del contrato de acciones.
FAILURE_FOLLOWUP: explanation_relation es relevante y debe respetar su enum.
CONVERSATION, PERSONAL_MEMORY_RECALL y controles CONFIRM/CANCEL/RETRY: omite metadata explicativa.
CLARIFICATION puede incluir explanation_relation para diagnosticar una referencia discursiva ambigua.
Si recibes branch_contract en repair, conserva el contexto original y corrige el error exacto
usando allowed_fields; no reintroduzcas inapplicable_fields. Nunca cambies autoridad para reparar forma.

intent es UNO de:
CREATE_NEW, EDIT_EXISTING, READ_EXISTING, REPLACE_EXISTING, EXECUTE_EXISTING,
DELETE_EXISTING, CONVERT_EXISTING, ARCHIVE_EXISTING, MOVE_EXISTING, RENAME_EXISTING,
CONFIRM, CANCEL, RETRY, RETAIN_UNVERIFIED,
EXPLAIN_FAILURE, CONTINUE_EXPLANATION, REMEDIATION_QUERY,
CONVERSATION, CONTENT_RESPONSE, PERSONAL_MEMORY_RECALL, PERSONAL_IDENTITY,
SELF_PREFERENCE, SELF_RELATIONAL, ENTITY_QUERY, PRE_REQUEST,
TOPIC_CLOSURE, OFFER_ACCEPTANCE, CLARIFICATION, UNSUPPORTED.

speech_act: ACTION_REQUEST, INFORMATION_REQUEST, SOCIAL o ACKNOWLEDGEMENT.
PRIMERO decide que esta solicitando AHORA el usuario, sin sustituirlo por el contexto anterior.
Una orden de ejecutar/probar un artifact es ACTION_REQUEST + EXECUTE_EXISTING aunque haya un fallo activo.
Una pregunta sobre la posibilidad de arreglar un fallo es INFORMATION_REQUEST + REMEDIATION_QUERY;
no es ejecutar. Una orden de corregir es EDIT_EXISTING/RETRY segun su objetivo real, no explicacion.
Para crear algo nuevo usa CREATE_NEW y el tipo solicitado; el foco no convierte creacion en edicion.
Para editar usa EDIT_EXISTING, para mostrar contenido READ_EXISTING (sin execute ni mutacion).
Para encargar pruebas declara TEST_SUITE, para solo ejecutar EXECUTION. Crear/editar sin pruebas usa BASIC.
Para toda peticion de codigo declara execution_intent: NONE si solo pide contenido,
RUN si solicita ejecucion y TEST si solicita pruebas. Evalua TODOS los objetivos del turno:
CREATE_NEW/EDIT_EXISTING pueden coexistir con RUN/TEST. Conserva ambos objetivos en primary_goal.
RUN/TEST requieren execute=true y verificacion EXECUTION/TEST_SUITE; nunca BASIC.
CREATE_NEW no requiere el target anterior; no pongas el nuevo filename en references como si ya existiera.
El objetivo completo, incluyendo nombre deseado y transformacion, va en primary_goal.

Relacion opcional: NEW_TOPIC, TOPIC_CONTINUATION, NEW_ACTION, ARTIFACT_FOLLOWUP,
NEW_CORRECTIVE_ACTION. Solo declarala si corresponde; el runtime deriva las relaciones de controles.
El failure topic sirve SOLO cuando el pedido actual se refiere semanticamente a ESE fallo.
No domina nuevas preguntas, valoraciones subjetivas, identidad, conocimiento externo ni nuevas ordenes.
No saber responder una pregunta no implica que el usuario pregunte por un error anterior.
La incertidumbre del classifier no es evidencia de FAILURE_FOLLOWUP.
EXPLAIN_FAILURE explica la causa; CONTINUE_EXPLANATION profundiza; REMEDIATION_QUERY evalua opciones.
Para esos tres controles usa el contrato MINIMO: solo intent, speech_act=INFORMATION_REQUEST,
primary_goal y confidence. NO incluyas execute, verification_requirement, turn_relation,
artifact_type ni references: el host deriva ausencia de efectos, relacion y failure_ref.
No heredes pruebas, verificacion ni operaciones de la tarea fallida: estas explicando, no actuando.
Ejemplo de contrato, no regla de frases:
{"intent":"EXPLAIN_FAILURE","speech_act":"INFORMATION_REQUEST","primary_goal":"Explicar la causa del fallo activo","confidence":0.85}
Si recibes allowed_explanation_contract y preserve_explanation en una reparacion,
conserva exactamente ese intent y el fallo indicado; omite los campos operativos y la relacion.
Usa previous_assistant_act junto con active_failure_topic: tras FAILURE_EXPLANATION,
una peticion referencial de desarrollar significado, implicaciones o detalle es
CONTINUE_EXPLANATION, no una necesidad de que el usuario aclare su objetivo.
SOLO en FAILURE_FOLLOWUP o en una CLARIFICATION que realmente intenta referirse a un fallo,
establece explanation_relation por la relacion semantica del turno con ESE fallo:
ASK_FAILURE_CAUSE, ELABORATE_PREVIOUS_EXPLANATION, ASK_REMEDIATION, NEW_ACTION,
TOPIC_CHANGE o AMBIGUOUS. Es metadata discursiva, no una operacion ni autoridad.
Para los controles explicativos puede acompanar el contrato minimo sin agregar efectos.
ELABORATE_PREVIOUS_EXPLANATION corresponde a CONTINUE_EXPLANATION;
ASK_FAILURE_CAUSE a EXPLAIN_FAILURE; ASK_REMEDIATION a REMEDIATION_QUERY.
Una nueva orden explicita o un tema distinto no son elaboracion, aunque haya un fallo activo.
Si eliges CLARIFICATION despues de FAILURE_EXPLANATION con fallo activo, declara
explanation_relation y ambiguities con el referente/objetivo realmente sin resolver.
No confundir no saber ampliar una respuesta con no entender que el usuario pide mas detalle.
Si recibes continuation_context, reevalua UNA vez esta relacion usando el turno original,
el acto previo y el fallo incluido. Puedes conservar una ambiguedad real con su motivo,
elegir nueva accion/cambio de tema, o el control explicativo adecuado. No ejecutes nada.
CANCEL cierra lo pendiente sin ejecutar. CONFIRM acepta la unica aprobacion pendiente, con alta certeza
si es destructiva. RETAIN_UNVERIFIED acepta incertidumbre de calidad, nunca permisos de seguridad.
Una oferta aceptada es OFFER_ACCEPTANCE solo si existe pending_offer.
Si pending_confirmation.available y previous_assistant_act=CONFIRMATION_REQUEST,
evalua PRIMERO la relacion del turno con ESA accion concreta (requested_action, artifact_names,
task_id, plan_id, permission_effects). Una aceptacion o instruccion de continuar esa accion
es intent=CONFIRM, speech_act=ACKNOWLEDGEMENT o ACTION_REQUEST,
pending_action_relation=ACCEPT_PENDING_ACTION. Mencionar la ejecucion pendiente no crea otra tarea.
Usa el contrato minimo de CONFIRM: intent, speech_act, primary_goal, confidence,
pending_action_relation. Omite referencias, execute, verification y artifact_type;
el host conserva EXACTAMENTE plan, TestSpec, bindings y permisos ya presentados.
Un rechazo es CANCEL + REJECT_PENDING_ACTION, sin ejecutar.
Un artifact/objetivo realmente diferente es NEW_ACTION y su operacion propia, NO CONFIRM.
Una respuesta realmente ambigua puede ser CLARIFICATION + AMBIGUOUS.
No fuerces aceptacion si la nueva intencion es distinta. No inferir una accion nueva
solo porque la aceptacion nombra la operacion que acaba de ofrecerse.
Si recibes pending_confirmation_contract, reevalua UNA vez esta relacion usando el
turno original y el resumen pendiente. Corrige los campos en conflicto sin agregar autoridad.

PRIMERO distingue el dominio de una pregunta no operacional:
- SELF_PREFERENCE: eleccion, opinion o inclinacion de Anydes, no un hecho sobre el usuario.
- SELF_RELATIONAL: valoracion subjetiva de Anydes SOBRE el interlocutor. No necesita
  una medicion objetiva ni una preferencia historica; usa impresiones del discurso y
  contexto respaldado, sin inventar recuerdos. No es pregunta causal sobre un fallo.
- PERSONAL_IDENTITY: identidad basica del interlocutor. Declara identity_attribute=DISPLAY_NAME
  si el atributo semantico consultado es su nombre de uso. Este campo es OBLIGATORIO
  en esta rama. Para un atributo realmente fuera del proveedor autorizado usa UNSUPPORTED;
  UNSPECIFIED o un campo ausente no son una resolucion final valida.
  El host consulta el contexto de usuario autorizado, no una busqueda amplia de recuerdos.
  Infiere QUE atributo se consulta por el significado completo del turno; nunca inventes SU VALOR.
  Si recibes identity_attribute_contract, la rama PERSONAL_IDENTITY ya esta decidida:
  conserva esa rama y INFORMATION_REQUEST, completa SOLO el atributo semantico pedido
  usando allowed_identity_attributes y el turno/contexto originales. No reclasifiques
  a memoria, SELF, failure ni accion; no agregues efectos ni valores de identidad.
- PERSONAL_MEMORY_RECALL: recuerdos/hechos compartidos por el usuario, aunque la consulta sea amplia.
Las tres primeras son INFORMATION_REQUEST, sin efectos, tools ni escritura de memoria.
Un dominio explicito nuevo prevalece sobre failure priors; puede continuar una charla
personal sin continuar el fallo. No emitas explanation_relation para esos dominios.
Una solicitud de responder/desarrollar una pregunta anterior usa recent_dialogue para
recuperar ESA pregunta; no exige automaticamente aclaracion ni implica un fallo.
Si memory_recall.active y el turno continua ese tema, conserva PERSONAL_MEMORY_RECALL;
no decidas que no hay memoria antes de recuperarla. Un cambio claro de tema termina ese proposito.
Preferencias de Anydes son SELF_PREFERENCE, no memoria del usuario ni una tool.
Conserva las entidades de discourse.subjects si el turno se refiere a ellas. No inventes hechos.
CLARIFICATION solo para ambiguedad real; no por no saber la solucion de un fallo.
Si recibes pre_request_contract en repair, reevalua SOLO la distincion entre apertura,
charla casual y solicitud concreta incompleta con el turno/contexto originales.
Corrige el conflicto entre intent y conversation_purpose, sin efectos ni invencion del tema.
No cambies una accion concreta por PRE_REQUEST para resolver un error de contrato.
No respondas al usuario, no concedas permisos, no inventes rutas ni ejecutes nada.
Si recibes repair_errors corrige exactamente el campo/constraint indicado sin cambiar la intencion.
'''


COMMON_FIELDS = frozenset({'intent', 'speech_act', 'primary_goal', 'confidence',
    'turn_relation', 'subjects', 'ambiguities', 'identity_attribute'})
OPERATIONAL_FIELDS = frozenset({'artifact_type', 'references', 'execute',
    'execution_intent', 'verification_requirement'})
FAILURE_FIELDS = frozenset({'explanation_relation'})
PENDING_FIELDS = frozenset({'pending_action_relation'})
DISCOURSE_FIELDS = frozenset({'conversation_purpose'})


def declared_branch(value):
    """A known discriminator is required before dropping any non-authority metadata."""
    from .operation_intent import ArtifactOperation
    if not isinstance(value, dict):
        return None
    intent, act = value.get('intent'), value.get('speech_act')
    if not isinstance(intent, str) or not isinstance(act, str):
        return None
    if act not in {'ACTION_REQUEST', 'INFORMATION_REQUEST', 'SOCIAL', 'ACKNOWLEDGEMENT'}:
        return None
    operations = {op.value for op in ArtifactOperation if op != ArtifactOperation.NONE}
    if intent in operations:
        return 'ARTIFACT_ACTION' if (act == 'ACTION_REQUEST' or
            intent == 'READ_EXISTING' and act == 'INFORMATION_REQUEST') else None
    if intent in {'EXPLAIN_FAILURE', 'CONTINUE_EXPLANATION', 'REMEDIATION_QUERY'}:
        return 'FAILURE_FOLLOWUP' if act == 'INFORMATION_REQUEST' else None
    if intent == 'CLARIFICATION':
        return 'DISCOURSE_CLARIFICATION'
    if intent in {'CONFIRM', 'CANCEL', 'RETRY', 'RETAIN_UNVERIFIED', 'OFFER_ACCEPTANCE'}:
        return 'TASK_CONTROL'
    if intent == 'PERSONAL_MEMORY_RECALL':
        return 'MEMORY_RECALL'
    if intent == 'PERSONAL_IDENTITY' and act == 'INFORMATION_REQUEST':
        return 'IDENTITY'
    if intent in {'CONVERSATION', 'CONTENT_RESPONSE', 'SELF_PREFERENCE', 'SELF_RELATIONAL',
                  'ENTITY_QUERY',
                  'TOPIC_CLOSURE', 'CAPABILITY_QUERY', 'UNSUPPORTED', 'PRE_REQUEST'}:
        return 'CONVERSATION'
    return None


def branch_contract(value, state=None):
    branch = declared_branch(value)
    relevant = FAILURE_FIELDS if branch in {'FAILURE_FOLLOWUP', 'DISCOURSE_CLARIFICATION'} else frozenset()
    context = (state or {}).get('resolution_context') or {}
    pending_relevant = ((context.get('pending_confirmation') or {}).get('available') or
        isinstance(value,dict) and value.get('intent') in {'CONFIRM','CANCEL'})
    if pending_relevant:
        relevant = relevant | PENDING_FIELDS
    if isinstance(value, dict) and value.get('intent') in {'CONVERSATION','PRE_REQUEST','CLARIFICATION'}:
        relevant = relevant | DISCOURSE_FIELDS
    # Operational fields are never disposable: even on another branch they must be
    # checked for effects/authority contradictions by the existing strict validator.
    return {'declared_branch':branch, 'allowed_fields':sorted(COMMON_FIELDS | OPERATIONAL_FIELDS | relevant),
        'required_fields':['identity_attribute'] if branch == 'IDENTITY' else [],
        'inapplicable_fields':sorted((FAILURE_FIELDS | PENDING_FIELDS | DISCOURSE_FIELDS) - relevant) if branch else [],
        'strict_authority_fields':sorted(OPERATIONAL_FIELDS | (PENDING_FIELDS if pending_relevant else frozenset()))}


def canonicalize_branch_fields(value, record=None, state=None):
    contract = branch_contract(value,state)
    if not contract['declared_branch']:
        return value
    removed = sorted(set(value) & set(contract['inapplicable_fields']))
    if not removed:
        return value
    canonical = {key:item for key,item in value.items() if key not in removed}
    if record is not None:
        record('interaction_branch_canonicalization', branch=contract['declared_branch'],
            removed_fields=removed, reason='INAPPLICABLE_NON_AUTHORITY_METADATA',
            repair_consumed=False)
    return canonical
