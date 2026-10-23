"""Bounded discourse evidence, never permission or execution authority."""
from dataclasses import asdict, dataclass, field
from enum import Enum
from uuid import uuid4


class ConversationPurpose(str, Enum):
    GENERAL = 'GENERAL'
    SELF_PREFERENCE = 'SELF_PREFERENCE'
    SELF_RELATIONAL = 'SELF_RELATIONAL'
    PERSONAL_IDENTITY = 'PERSONAL_IDENTITY'
    PERSONAL_MEMORY = 'PERSONAL_MEMORY'
    ENTITY_QUERY = 'ENTITY_QUERY'
    ACKNOWLEDGEMENT = 'ACKNOWLEDGEMENT'
    PRE_REQUEST = 'PRE_REQUEST'


class IdentityAttribute(str, Enum):
    UNSPECIFIED = 'UNSPECIFIED'
    DISPLAY_NAME = 'DISPLAY_NAME'
    UNSUPPORTED = 'UNSUPPORTED'


@dataclass(frozen=True)
class DiscourseEntity:
    name: str
    semantic_type: str

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or set(value) != {'name', 'semantic_type'}:
            raise ValueError('discourse_entity_schema')
        if any(not isinstance(value[k], str) or not value[k].strip() or len(value[k]) > 160
               for k in ('name', 'semantic_type')):
            raise ValueError('discourse_entity_value')
        return cls(**value)


@dataclass
class MemoryRecallContext:
    active: bool = False
    purpose: str = 'PERSONAL_MEMORY_RECALL'
    subject: str = 'USER'
    sensitivity_scope: str = 'NON_SENSITIVE'
    used_evidence_ids: list[str] = field(default_factory=list)
    result_state: str = ''


@dataclass
class DiscourseContext:
    subjects: tuple[DiscourseEntity, ...] = ()
    purpose: ConversationPurpose = ConversationPurpose.GENERAL
    turns: list[dict] = field(default_factory=list)
    pending_offer: dict | None = None
    memory_recall: MemoryRecallContext = field(default_factory=MemoryRecallContext)

    @property
    def memory_evidence_used(self):
        return self.memory_recall.used_evidence_ids

    def snapshot(self, *, include_turns=False):
        offer = self.pending_offer
        if offer and not include_turns:
            offer = {k:offer[k] for k in ('offer_id','operation_kind')}
        result = {'subjects': [asdict(e) for e in self.subjects], 'purpose': self.purpose.value,
                  'pending_offer':offer, 'memory_recall':asdict(self.memory_recall)}
        if include_turns:
            result['recent_turns'] = self.turns[-6:]
        return result

    def resolve(self, decision):
        if decision.turn_relation.value in {'NEW_TOPIC', 'TOPIC_CLOSURE'}:
            self.subjects = ()
            self.pending_offer = None
        if decision.subjects:
            self.subjects = decision.subjects
        self.purpose = decision.conversation_purpose
        self.memory_recall.active = self.purpose == ConversationPurpose.PERSONAL_MEMORY

    def record(self, user, assistant):
        self.turns.extend([{'role':'user', 'content':str(user)[:2400]},
                           {'role':'assistant', 'content':str(assistant)[:2400]}])
        del self.turns[:-12]


def supported_offer(value, registry, contracts):
    """Only an enabled, locally grounded semantic operation can be offered.

    This is preflight plausibility, not an executable plan or a permission grant.
    Acceptance still goes through normal planning, validation and confirmation.
    """
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {'request', 'operation_kind'}
            or not isinstance(value['request'], str) or not 1 <= len(value['request']) <= 400):
        raise ValueError('offer_schema_invalid')
    enabled = set(registry.list_capabilities(enabled_only=True))
    compatible = [c for c in contracts if c['operation_kind'] == value['operation_kind']
                  and all(cap in enabled for cap in (c['capability'], *c.get('requires', ())))]
    if not compatible:
        return None
    return {'offer_id':'offer:'+uuid4().hex, **value,
            'question':'Puedo preparar esta accion: '+value['request'].rstrip('.?')+'. Quieres que lo haga?'}
