from __future__ import annotations

from anydes.identity.relationship_contracts import RelationshipSnapshot
from anydes.identity.relationship_repository import load_relationship_snapshot
from anydes.identity.relationship_store import RelationshipStateStore
from anydes.nucleo.config import Config


class RelationshipProvider:
    """Provides shared-history state without deciding or modifying anything."""

    def __init__(
        self,
        config: Config,
        store: RelationshipStateStore | None = None,
    ) -> None:
        self.config = config
        self.store = store or RelationshipStateStore.from_config(config)

    def snapshot(self) -> RelationshipSnapshot:
        return load_relationship_snapshot(self.config, self.store)
