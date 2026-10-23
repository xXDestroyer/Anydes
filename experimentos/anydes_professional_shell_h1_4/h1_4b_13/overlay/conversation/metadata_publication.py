"""Bind already-resolved metadata to prose; never select or operate on artifacts."""
from dataclasses import replace
import hashlib
import re
from .publication_boundary import PublicationFailure


class MetadataPublicationError(PublicationFailure):
    def __init__(self,reason,field='outcome_facts.value'):
        self.field_path=field
        super().__init__(reason,'METADATA_BINDING')


def bound_facts(controller,decision,facts):
    """Check the selected identity, not another search or discourse resolution."""
    facts=dict(facts)
    if facts.get('metadata_status')!='AVAILABLE':
        facts.pop('value',None)
        facts['location_authorized']=False
        return facts
    index=controller.state.recent_artifact_state
    item=next((row for row in index.artifacts if row.artifact_id==facts.get('evidence_id')),None)
    ref=item.runtime_ref if item else None
    if (ref is None or facts.get('requested_field')!=decision.metadata_field
            or facts.get('version')!=ref.version):
        raise MetadataPublicationError('METADATA_EVIDENCE_BINDING_MISMATCH','outcome_facts.evidence_id')
    references=set(decision.references)
    if ((references and not references.intersection({ref.artifact_id,ref.display_name}))
            or (not references and ref.artifact_id not in index.focus_artifact_ids)):
        raise MetadataPublicationError('METADATA_EVIDENCE_BINDING_MISMATCH','outcome_facts.evidence_id')
    values={'LOCATION':ref.path,'NAME':ref.display_name,'TYPE':ref.artifact_type,
        'VERSION':ref.version,'CREATED_AT':item.created_at,'CURRENT_ARTIFACT':ref.display_name}
    expected=values.get(decision.metadata_field)
    if expected is None or isinstance(expected,str) and not expected.strip():
        facts.pop('value',None)
        facts.update(metadata_status='UNAVAILABLE',location_authorized=False)
        return facts
    if facts.get('value')!=expected:
        raise MetadataPublicationError('METADATA_EVIDENCE_BINDING_MISMATCH')
    if decision.metadata_field=='LOCATION' and facts.get('location_authorized') is not True:
        facts.pop('value',None)
        facts.update(metadata_status='RESTRICTED',location_authorized=False)
    facts['claim_scope']='EXISTING_ARTIFACT_METADATA_ONLY'
    return facts


def opaque_value(text,facts):
    """Exact factual bytes are data, not punctuation for prose normalization."""
    from .publication import _PRIVATE_PATH
    if facts.get('metadata_status')!='AVAILABLE':
        if _PRIVATE_PATH.search(text):
            raise MetadataPublicationError('METADATA_UNAVAILABLE_PATH_CLAIM','response.text')
        return text,False
    value=facts.get('value')
    if value is None or isinstance(value,bool) or not str(value).strip() or not facts.get('evidence_id'):
        raise MetadataPublicationError('METADATA_VALUE_UNAVAILABLE')
    literal=str(value)
    pattern=re.escape(literal)
    if isinstance(value,(int,float)):
        pattern=r'(?<![\w.])'+pattern+r'(?!\w|\.\d)'
    else:
        pattern=r'(?<![\w\\/])'+pattern+r'(?![\w\\/]|\.\w)'
    token='AUTHMETADATA'+hashlib.sha256(literal.encode()).hexdigest()[:16]
    if token in text:
        raise MetadataPublicationError('METADATA_TOKEN_COLLISION','response.text')
    masked,count=re.subn(pattern,lambda _:token,text)
    if not count:
        raise MetadataPublicationError('METADATA_VALUE_NOT_PRESERVED','response.text')
    if _PRIVATE_PATH.search(masked):
        raise MetadataPublicationError('UNAUTHORIZED_METADATA_LOCATION','response.text')
    return masked,True


def preserve_factual_prose(local,masked,original):
    # Sentence-joining whitespace is not a factual rewrite. All other edits and
    # blocking issues still require the ordinary bounded wording repair.
    if local.valid and ' '.join(local.text.split())==' '.join(masked.split()):
        return replace(local,text=original)
    return local
