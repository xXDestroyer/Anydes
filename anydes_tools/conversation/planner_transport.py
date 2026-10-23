"""Bounded planner use of the existing primary/local clients, not chat continuation."""
from .plan_protocol import PlanProtocolError


def planner_stream(service, messages, *, state, record, max_output_tokens):
    from anydes.hybrid.router import ProviderFailure

    router = getattr(service.transport, 'router', None)
    if router is None:
        # Non-hybrid configured transports still use the same strict parser.
        return service.transport.stream(messages, max_output_tokens=max_output_tokens)
    outbound, privacy = router.sanitizer.sanitize(messages)
    record('planner_transport_context', provider=state.get('provider', 'primary'),
        messages=len(outbound), characters=privacy.characters_after,
        redactions=privacy.redactions, repair_used=state.get('repair_used', False))
    if state.get('provider', 'primary') == 'primary':
        record('planner_provider_attempt', provider='primary', model=router.online.model)
        try:
            outcome = router.online.stream(outbound, max_output_tokens=max_output_tokens)
            if outcome.connection_state == 'completed' and not outcome.error_type:
                return outcome
            reason = outcome.error_type or outcome.interruption_reason or 'interrupted'
        except ProviderFailure as error:
            reason = error.error_type
            record('planner_provider_unavailable', provider='primary',
                status=error.status, reason_code=reason)
        state['provider'] = 'fallback'
        record('planner_provider_fallback', reason_code=reason, retry_policy='NO_INLINE_RETRY_ONE_FALLBACK')
    # Request-local settings avoid mutating the conversational transport or model.
    # A cold 8B load is bounded at 90s, instead of returning empty after preload wait.
    config = router.config
    local = router.local
    transport = type(local.transport)(base_url=local.base_url, model=local.model,
        keep_alive=config.local_keep_alive, timeout_seconds=90.0,
        max_output_tokens=max_output_tokens, num_ctx=max(8192, config.local_num_ctx))
    record('planner_provider_attempt', provider='fallback', model=local.model,
        timeout_seconds=90, num_ctx=max(8192, config.local_num_ctx))
    try:
        outcome = transport.stream(outbound, max_output_tokens=max_output_tokens)
    except (ProviderFailure, OSError, ValueError) as error:
        record('planner_provider_unavailable', provider='fallback', reason_code=type(error).__name__)
        raise PlanProtocolError('PLANNER_PROVIDERS_UNAVAILABLE') from error
    if outcome.connection_state != 'completed' or outcome.error_type:
        record('planner_provider_unavailable', provider='fallback',
            reason_code=outcome.error_type or outcome.interruption_reason or 'interrupted')
        raise PlanProtocolError('PLANNER_PROVIDERS_UNAVAILABLE')
    return outcome
