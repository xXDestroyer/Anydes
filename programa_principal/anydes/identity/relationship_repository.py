from __future__ import annotations

from anydes.identity.relationship_contracts import RelationshipSnapshot
from anydes.identity.relationship_store import RelationshipStateError, RelationshipStateStore
from anydes.nucleo.config import Config


def load_relationship_snapshot(
    config: Config,
    store: RelationshipStateStore | None = None,
) -> RelationshipSnapshot:
    """Read relational state without mutating or deciding how it should be used."""
    repository = store or RelationshipStateStore.from_config(config)
    try:
        document = repository.read()
        return RelationshipSnapshot.from_collections(
            revision=document.revision,
            updated_at=document.updated_at,
            collections=document.collections,
        )
    except (RelationshipStateError, KeyError, TypeError, ValueError) as exc:
        return RelationshipSnapshot(
            available=False,
            warnings=(f"relationship_state_unavailable:{type(exc).__name__}",),
        )
