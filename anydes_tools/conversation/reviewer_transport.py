"""Request-local structured review, never a conversational continuation."""
import hashlib
import json
from .diagnostics import boundary


class ReviewerUnavailable(RuntimeError):
    pass


class ReviewerTransport:
    """One primary attempt, one fallback, at most one protocol repair by caller."""

    def __init__(self, transport, stream, record):
        self.router = getattr(transport, 'router', None)
        self.stream = stream
        self.record = record
        self.local = None

    def prepare(self, messages):
        if self.router is not None:
            messages, privacy = self.router.sanitizer.sanitize(messages)
            self.record('reviewer_context', messages=len(messages),
                        redactions=privacy.redactions, characters=privacy.characters_after)
        return messages

    def attempt(self, messages, *, provider, repair=False):
        from anydes.hybrid.router import ProviderFailure
        digest = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        if self.router is None:
            if provider == 'fallback':
                raise ReviewerUnavailable('NO_CONFIGURED_REVIEWER_FALLBACK')
            client = None
            stream = self.stream
        elif provider == 'primary':
            client = self.router.online
            stream = client.stream
        else:
            if self.local is None:
                local, config = self.router.local, self.router.config
                # Bypass the chat-only fallback note and its tiny conversational context.
                self.local = type(local.transport)(base_url=local.base_url, model=local.model,
                    keep_alive=config.local_keep_alive, timeout_seconds=90.0,
                    max_output_tokens=384, num_ctx=max(8192, config.local_num_ctx))
            client = self.local
            stream = client.stream
        self.record('reviewer_provider_attempt', provider=provider, repair=repair,
                    model=getattr(client, 'model', 'configured_transport'),
                    request_sha256=digest, structured_contract=True,
                    conversation_wrapper=False, max_output_tokens=384)
        try:
            with boundary('REVIEWER_PROVIDER_' + provider.upper()):
                result = stream(messages, on_content=None, max_output_tokens=384)
        except (ProviderFailure, OSError) as error:
            self.record('reviewer_provider_unavailable', provider=provider,
                        status=getattr(error, 'status', 0), reason_code=type(error).__name__)
            raise ReviewerUnavailable('REVIEWER_PROVIDER_UNAVAILABLE') from error
        if (getattr(result, 'connection_state', 'completed') != 'completed'
                or getattr(result, 'error_type', '')):
            self.record('reviewer_provider_unavailable', provider=provider, status=0,
                        reason_code=getattr(result, 'error_type', '') or 'INCOMPLETE')
            raise ReviewerUnavailable('REVIEWER_RESPONSE_INCOMPLETE')
        return result
