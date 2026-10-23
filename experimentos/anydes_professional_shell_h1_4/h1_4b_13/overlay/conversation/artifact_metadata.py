"""Read-only projection of committed metadata, with no filesystem action."""
from enum import Enum


class MetadataField(str, Enum):
    LOCATION = 'LOCATION'
    NAME = 'NAME'
    TYPE = 'TYPE'
    VERSION = 'VERSION'
    CREATED_AT = 'CREATED_AT'
    CURRENT_ARTIFACT = 'CURRENT_ARTIFACT'


def metadata_evidence(controller, decision):
    index = controller.state.recent_artifact_state
    requested = set(decision.references)
    candidates = [item for item in index.artifacts if index._selectable(item) and item.runtime_ref]
    if requested:
        candidates = [item for item in candidates if
            item.artifact_id in requested or item.filename in requested]
    else:
        candidates = [item for item in candidates if item.artifact_id in index.focus_artifact_ids]
    if len(candidates) != 1:
        return {'metadata_status':'AMBIGUOUS' if len(candidates)>1 else 'NOT_FOUND',
            'requested_field':decision.metadata_field,
            'candidate_names':[item.filename for item in candidates]}
    item = candidates[0]
    ref = item.runtime_ref
    policy = controller.boundary.classify(ref.path, operation='READ',
        authorized_scopes=controller.state.authorized_scopes)
    if not policy.allowed:
        return {'metadata_status':'RESTRICTED', 'requested_field':decision.metadata_field}
    values = {'LOCATION':ref.path, 'NAME':ref.display_name, 'TYPE':ref.artifact_type,
        'VERSION':ref.version, 'CREATED_AT':item.created_at,
        'CURRENT_ARTIFACT':ref.display_name}
    return {'metadata_status':'AVAILABLE', 'requested_field':decision.metadata_field,
        'value':values[decision.metadata_field], 'display_name':ref.display_name,
        'version':ref.version, 'evidence_id':ref.artifact_id,
        'location_authorized':decision.metadata_field == 'LOCATION'}
