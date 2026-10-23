"""Semantic requirement provenance, not execution authority or personal memory."""
from dataclasses import dataclass, field
from uuid import uuid4


def parse_update(value):
    if value is None:
        return None
    if not isinstance(value,dict) or set(value)-{'relation','operation','artifact_type','requirements','clarification_target','task_id','requirement_items','planning_readiness'}:
        raise ValueError('task_update_schema')
    if 'task_id' in value and (not isinstance(value['task_id'],str) or not value['task_id'] or len(value['task_id'])>128):
        raise ValueError('task_update_task_id_invalid')
    if value.get('relation') not in {'START','CONTINUE','REVISE','SWITCH'}:
        raise ValueError('task_update_relation')
    if value.get('operation','CREATE_NEW') not in {'CREATE_NEW','EDIT_EXISTING'}:
        raise ValueError('task_update_operation')
    if value.get('artifact_type','CODE') not in {'CODE','DOCUMENT','TEXT'}:
        raise ValueError('task_update_artifact_type')
    rows=value.get('requirements',[])
    if not isinstance(rows,list) or len(rows)>24:
        raise ValueError('task_requirements_bounds')
    for row in rows:
        if not isinstance(row,dict) or set(row) != {'key','value'} or any(
                not isinstance(row[k],str) or not row[k].strip() or len(row[k])>(100 if k=='key' else 1000)
                for k in ('key','value')):
            raise ValueError('task_requirement_schema')
    if len({r['key'] for r in rows}) != len(rows):
        raise ValueError('task_requirement_duplicate')
    target=value.get('clarification_target')
    if target is not None and (not isinstance(target,dict) or set(target)!={'key','question'} or any(
            not isinstance(target[k],str) or not target[k].strip() or len(target[k])>500 for k in target)):
        raise ValueError('clarification_target_schema')
    items=value.get('requirement_items',[])
    if not isinstance(items,list) or len(items)>24:
        raise ValueError('requirement_items_bounds')
    for item in items:
        if (not isinstance(item,dict) or set(item)-{'key','necessity','question','default_value'}
                or not {'key','necessity'} <= item.keys()
                or item['necessity'] not in {'REQUIRED','OPTIONAL','DEFAULTABLE'}
                or not isinstance(item['key'],str) or not item['key'].strip() or len(item['key'])>100):
            raise ValueError('requirement_item_schema')
        for name in ('question','default_value'):
            if name in item and (not isinstance(item[name],str) or not item[name].strip() or len(item[name])>1000):
                raise ValueError('requirement_item_value')
        if item['necessity']=='DEFAULTABLE' and 'default_value' not in item:
            raise ValueError('defaultable_requires_semantic_default')
        if item['necessity']=='REQUIRED' and 'default_value' in item:
            raise ValueError('required_cannot_be_defaulted')
    if len({i['key'] for i in items})!=len(items):
        raise ValueError('requirement_item_duplicate')
    readiness=value.get('planning_readiness')
    if readiness is not None and (not isinstance(readiness,dict)
            or set(readiness)!={'goal_resolvable','artifact_specified','material_contradictions'}
            or type(readiness['goal_resolvable']) is not bool or type(readiness['artifact_specified']) is not bool
            or not isinstance(readiness['material_contradictions'],list)
            or len(readiness['material_contradictions'])>24
            or any(not isinstance(k,str) or not k.strip() or len(k)>100 for k in readiness['material_contradictions'])):
        raise ValueError('planning_readiness_schema')
    return value


@dataclass
class TaskContract:
    task_id: str
    goal: str
    operation: str
    artifact_type: str
    execution_required: bool = False
    verification_requirement: str = 'BASIC'
    requirements: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    unresolved_questions: dict = field(default_factory=dict)
    resolved_targets: set = field(default_factory=set)
    suspended: bool = False
    requirement_items: dict = field(default_factory=dict)
    goal_resolvable: bool = True
    artifact_specified: bool = True
    material_contradictions: list = field(default_factory=list)
    _readiness_diagnostics: list = field(default_factory=list,repr=False)

    def __post_init__(self):
        self.recompute_resolvability(check_stale=False)

    @classmethod
    def start(cls, decision, update):
        return cls('task:'+uuid4().hex,decision.primary_goal,
            update.get('operation',decision.operation_intent.value if decision.operation_intent.value!='NO_ARTIFACT_ACTION' else 'CREATE_NEW'),
            update.get('artifact_type','CODE'),decision.requires_execution,
            decision.verification_requirement.value if decision.verification_requirement.value!='NONE' else 'BASIC')

    def merge(self, update, turn_id):
        parse_update(update)
        validate_continuity(update,{'active_task':{'semantic_contract':self.snapshot()}})
        for item in update.get('requirement_items',[]):
            previous=self.requirement_items.get(item['key'])
            if previous and previous['necessity']!=item['necessity'] and update['relation']!='REVISE':
                raise ValueError('requirement_necessity_revision_required')
            self.requirement_items[item['key']]=dict(item)
            if item['necessity']!='REQUIRED':
                self.unresolved_questions.pop(item['key'],None)
        readiness=update.get('planning_readiness')
        if readiness:
            self.artifact_specified=readiness['artifact_specified']
            self.material_contradictions=list(readiness['material_contradictions'])
        for row in update.get('requirements',[]):
            key,value=row['key'],row['value']
            old=self.requirements.get(key)
            if old is not None and old['value'] != value and update['relation']!='REVISE':
                raise ValueError('task_requirement_conflict_requires_revision')
            if old is not None and old['value'] != value:
                for entry in self.history:
                    if entry['key']==key and not entry['superseded']:
                        entry['superseded']=True
            if old is None or old['value'] != value:
                entry={'key':key,'value':value,'turn_id':turn_id,'superseded':False}
                self.history.append(entry)
                self.requirements[key]=dict(entry)
            self.unresolved_questions.pop(key,None)
            self.resolved_targets.add(key)
        target=update.get('clarification_target')
        if target:
            if target['key'] in self.resolved_targets:
                raise ValueError('clarification_target_already_resolved')
            self.unresolved_questions[target['key']]=target['question']
            self.requirement_items.setdefault(target['key'],{'key':target['key'],
                'necessity':'REQUIRED','question':target['question']})
        for key,item in self.requirement_items.items():
            if key in self.requirements:
                continue
            if item['necessity']=='REQUIRED':
                if not item.get('question'):
                    raise ValueError('required_unresolved_question_missing')
                self.unresolved_questions.setdefault(key,item['question'])
            elif item['necessity']=='DEFAULTABLE':
                entry={'key':key,'value':item['default_value'],'turn_id':turn_id,
                    'superseded':False,'source':'SEMANTIC_DEFAULT'}
                self.history.append(entry)
                self.requirements[key]=dict(entry)

        self.recompute_resolvability(check_stale=False)
        if readiness and readiness['goal_resolvable']!=self.goal_resolvable:
            self._readiness_diagnostics.append({'reason_code':'MODEL_RESOLVABILITY_ASSESSMENT_OVERRIDDEN',
                'reported':readiness['goal_resolvable'],'derived':self.goal_resolvable})

    def recompute_resolvability(self, *, check_stale=True):
        """The validated semantic task, not an LLM boolean, determines readiness."""
        unresolved={key:question for key,question in self.unresolved_questions.items()
            if key not in self.requirements and
            self.requirement_items.get(key,{}).get('necessity','REQUIRED')=='REQUIRED'}
        for key,item in self.requirement_items.items():
            if item['necessity']=='REQUIRED' and key not in self.requirements:
                unresolved.setdefault(key,item.get('question',''))
        self.unresolved_questions=unresolved
        reasons=[]
        if not self.goal.strip(): reasons.append('GOAL_NOT_SPECIFIED')
        if self.operation not in {'CREATE_NEW','EDIT_EXISTING'}:
            reasons.append('GOAL_OPERATION_UNSPECIFIED')
        if (not self.artifact_specified or self.artifact_type not in {'CODE','DOCUMENT','TEXT'}):
            reasons.append('ARTIFACT_INTENT_UNRESOLVED')
        if unresolved: reasons.append('REQUIRED_REQUIREMENT_UNRESOLVED')
        if self.material_contradictions: reasons.append('MATERIAL_CONTRADICTION_UNRESOLVED')
        derived=not reasons
        if check_stale and self.goal_resolvable is False and derived:
            self._readiness_diagnostics.append({'reason_code':'STALE_GOAL_RESOLVABILITY_STATE',
                'previous':False,'derived':True,'recomputed':True})
        self.goal_resolvable=derived
        return reasons

    def record_resolvability(self,record,reasons):
        facts={'task_id':self.task_id,'resolved_requirements':sorted(self.requirements),
            'unresolved_requirements':sorted(self.unresolved_questions),
            'blocking_contradictions':list(self.material_contradictions),
            'artifact_intent_state':{'operation':self.operation,'artifact_type':self.artifact_type,
                'specified':self.artifact_specified},'goal_understood':bool(self.goal.strip()),
            'goal_resolvable':self.goal_resolvable,
            'blockers':[{'reason_code':code} for code in reasons],
            'values_logged':False}
        for detail in self._readiness_diagnostics:
            record('pending_task_validation_error' if detail['reason_code']=='STALE_GOAL_RESOLVABILITY_STATE'
                else 'pending_task_resolvability_assessment',**facts,**detail)
        self._readiness_diagnostics.clear()
        record('pending_task_resolvability',**facts,
            reason_code=reasons[0] if reasons else 'GOAL_RESOLVABLE')

    def planning_gate(self,record=None):
        reasons=self.recompute_resolvability()
        if record is not None:
            self.record_resolvability(record,reasons)
        if self.execution_required != (self.verification_requirement in {'EXECUTION','TEST_SUITE'}):
            reasons.append('EXECUTION_VERIFICATION_INCONSISTENT')
        return {'ready':not reasons,'reason_codes':reasons,
            'required_unresolved':list(self.unresolved_questions),
            'next_target':next(iter(self.unresolved_questions),None)}

    def snapshot(self):
        gate=self.planning_gate()
        return {'task_id':self.task_id,'goal':self.goal,'operation':self.operation,
            'artifact_type':self.artifact_type,'execution_required':self.execution_required,
            'verification_requirement':self.verification_requirement,
            'requirements':list(self.requirements.values()),
            'requirement_states':{key:'RESOLVED' for key in self.requirements},
            'unresolved_questions':dict(self.unresolved_questions),
            'resolved_targets':sorted(self.resolved_targets),'suspended':self.suspended,
            'requirement_items':list(self.requirement_items.values()),
            'required_resolved':[k for k in self.requirements if self.requirement_items.get(k,{}).get('necessity','REQUIRED')=='REQUIRED'],
            'required_unresolved':list(self.unresolved_questions),
            'optional':[k for k,i in self.requirement_items.items() if i['necessity']=='OPTIONAL'],
            'defaultable':[k for k,i in self.requirement_items.items() if i['necessity']=='DEFAULTABLE'],
            'planning_readiness':{'goal_resolvable':self.goal_resolvable,'artifact_specified':self.artifact_specified,
                'material_contradictions':list(self.material_contradictions)},
            'planning_gate':gate}


def validate_continuity(update, context):
    parse_update(update)
    if update:
        pending=(context.get('active_task') or {}).get('semantic_contract') or {}
        continuing=update['relation'] in {'CONTINUE','REVISE'}
        resolved={r['key'] for r in pending.get('requirements',[])} if continuing else set()
        resolved.update(r['key'] for r in update.get('requirements',[]))
        existing={i['key']:i for i in pending.get('requirement_items',[])} if continuing else {}
        for item in update.get('requirement_items',[]):
            previous=existing.get(item['key'])
            if previous and previous['necessity']!=item['necessity'] and update['relation']!='REVISE':
                raise ValueError('requirement_necessity_revision_required')
            if (item['necessity']=='REQUIRED' and item['key'] not in resolved
                    and not item.get('question')):
                raise ValueError('required_unresolved_question_missing')
    if update and update['relation'] in {'CONTINUE','REVISE'}:
        pending=(context.get('active_task') or {}).get('semantic_contract') or {}
        if not pending or pending.get('suspended'):
            raise ValueError('task_continuation_without_pending_contract')
        if (context.get('active_task') or {}).get('status') in {
                'COMPLETED','COMPLETED_PARTIAL','CANCELLED','SUPERSEDED','EXPIRED','IDLE'}:
            raise ValueError('task_continuation_requires_pending_task')
        task_id=pending['task_id']
        if (update.get('task_id',task_id)!=task_id or
                (context.get('active_task') or {}).get('task_id',task_id)!=task_id):
            raise ValueError('task_update_task_id_mismatch')
        if (context.get('pending_confirmation') or {}).get('available'):
            raise ValueError('task_update_cannot_replace_pending_confirmation')
        unresolved=pending.get('unresolved_questions') or {}
        if unresolved and not any(row['key'] in unresolved for row in update.get('requirements',[])):
            raise ValueError('pending_clarification_answer_required')
        known={r['key']:r['value'] for r in pending.get('requirements',[])}
        for row in update.get('requirements',[]):
            if row['key'] in known and known[row['key']]!=row['value'] and update['relation']!='REVISE':
                raise ValueError('task_requirement_conflict_requires_revision')
        target=update.get('clarification_target')
        if target and target['key'] in pending.get('resolved_targets',[]):
            raise ValueError('clarification_target_already_resolved')


def accept_update(controller, decision, turn_id):
    update=decision.task_update
    current=controller.state.pending_task
    if not update:
        if decision.turn_relation.value in {'NEW_TOPIC','TOPIC_CLOSURE','NEW_ACTION'} and current.semantic_contract:
            current.semantic_contract.suspended=True
        return
    if update['relation'] in {'START','SWITCH'}:
        contract=TaskContract.start(decision,update)
    else:
        contract=current.semantic_contract
        if contract is None or contract.suspended:
            raise ValueError('task_continuation_without_pending_contract')
    contract.merge(update,turn_id)
    current.semantic_contract=contract
    current.task_id=contract.task_id
    current.status='PLANNING' if contract.planning_gate(record=controller.audit.record)['ready'] else 'AWAITING_CLARIFICATION'
    controller.audit.record('task_contract_updated',task_id=contract.task_id,turn_id=turn_id,
        requirement_keys=sorted(contract.requirements),
        unresolved_keys=sorted(contract.unresolved_questions),
        resolved_keys=sorted(contract.resolved_targets),
        provenance_turn_ids=[r['turn_id'] for r in contract.history],values_logged=False)


def diagnostic_context(context):
    """Do not duplicate requirement values in context audit events."""
    from copy import deepcopy
    result=deepcopy(context)
    task=result.get('active_task') or {}
    contract=task.get('semantic_contract')
    if contract:
        contract.pop('goal',None)
        contract['requirements']=[{k:v for k,v in row.items() if k!='value'} for row in contract['requirements']]
        contract['unresolved_questions']=list(contract['unresolved_questions'])
        contract['requirement_items']=[{k:v for k,v in row.items() if k not in {'question','default_value'}}
            for row in contract.get('requirement_items',[])]
    return result


def continuation_repair_contract(context):
    """Same task/branch; the model completes only the semantic delta."""
    pending=(context.get('active_task') or {}).get('semantic_contract') or {}
    return {'intent':'CONTINUE_TASK','task_id':pending.get('task_id',''),
        'goal':pending.get('goal',''),
        'pending_requirements':pending.get('requirements',[]),
        'unresolved_questions':pending.get('unresolved_questions',{}),
        'resolved_targets':pending.get('resolved_targets',[]),
        'previous_assistant_act':context.get('previous_assistant_act'),
        'required_fields':['intent','speech_act','primary_goal','confidence','task_update'],
        'allowed_update_fields':['task_id','relation','requirements','clarification_target','requirement_items','planning_readiness'],
        'allowed_update_relations':['CONTINUE','REVISE'],
        'forbidden_repair_intents':['CONFIRM','CREATE_NEW'],
        'constraint':'Preserve CONTINUE_TASK and task_id. Complete only the semantic delta from the original user answer. Use the exact unresolved concept key when answering a clarification. Never invent a value. Empty requirements are valid only for a genuine resume with no unresolved question. If meaning cannot be established, leave it unresolved; do not invent confirmation or a new task.'}
