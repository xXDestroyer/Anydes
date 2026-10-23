"""Nonempty publication is authority, independent of generated prose style."""
from dataclasses import replace


class PublicationFailure(ValueError):
    def __init__(self, reason='EMPTY_USER_VISIBLE_RESPONSE', stage='PUBLICATION'):
        self.reason_code, self.stage = reason, stage
        super().__init__(reason)


def require_visible(final):
    if not isinstance(final.visible_text, str) or not final.visible_text.strip():
        raise PublicationFailure()
    if not isinstance(final.spoken_text, str) or not final.spoken_text.strip():
        raise PublicationFailure('EMPTY_SPOKEN_RESPONSE')
    if final.visible_text != final.spoken_text:
        raise PublicationFailure('CHAT_VOICE_PAYLOAD_MISMATCH')
    return final


def recover_publication(assistant, controller, outcome, final, user_turn, failure):
    """One separate failure-response request, never unreviewed original content."""
    from .outcome_voice import OutcomeVoiceRequest, ResponseIntent, generate_outcome_voice
    assistant.audit.record('publication_rejected', reason_code=failure.reason_code,
        stage=failure.stage, task_id=outcome.task_id, published=False)
    request = OutcomeVoiceRequest(ResponseIntent.EXPLAIN_FAILURE, user_turn,
        outcome.user_goal, (), {
            'status':'FAILED', 'failure_type':failure.reason_code,
            'stage':failure.stage, 'recoverable':True,
            'cause_detail_present':True,
            'recorded_failure':'La respuesta no pudo aprobarse para mostrarse; no se conoce una causa adicional.',
            'success_claim_allowed':False,
            'execution_observed':outcome.safe_evidence.get('code_executed'),
            'original_operation_status':outcome.status.value,
            'recoverability_scope':'RESPONSE_ONLY_NEVER_REPEAT_ACTION',
        }, {})
    try:
        text = generate_outcome_voice(assistant, request)
    except (OSError, ValueError, RuntimeError) as error:
        raise PublicationFailure('PUBLICATION_RECOVERY_UNAVAILABLE', 'RECOVERY') from error
    recovered = require_visible(replace(final, visible_text=text, spoken_text=text,
        content_segments=(), sanitized=True))
    assistant.audit.record('publication_recovered', reason_code=failure.reason_code,
        same_chat_voice=True, grounded=True, task_id=outcome.task_id)
    return recovered
