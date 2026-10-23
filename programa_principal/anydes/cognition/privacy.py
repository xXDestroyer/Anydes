from __future__ import annotations

from dataclasses import replace

from anydes.cognition.contracts import InternalContext, MemorySnapshot, Observation
from anydes.identity import IdentityCore, RelationshipModel, RelationshipSnapshot
from anydes.nucleo.interlocutor import InterlocutorContext


def context_for_interlocutor(
    context: InternalContext,
    interlocutor: InterlocutorContext,
) -> InternalContext:
    """Apply the privacy boundary before context reaches reasoning."""

    observation = Observation(
        id=f"obs:interlocutor:{interlocutor.state}",
        source="interlocutor",
        summary=_observation_summary(interlocutor),
        confidence=interlocutor.confidence if interlocutor.state == "verified_voice" else 1.0,
        salience=1.0,
        data={
            "state": interlocutor.state,
            "channel": interlocutor.channel,
            "owner_verified": interlocutor.owner_verified,
            "personal_memory_available": interlocutor.allows_personal_memory,
        },
    )
    observations = tuple(item for item in context.observations if item.source != "interlocutor") + (
        observation,
    )
    if interlocutor.allows_personal_memory:
        return replace(context, observations=observations)
    self_model = _self_without_relationship(context.self_model, interlocutor)
    return replace(
        context,
        observations=observations,
        memory=MemorySnapshot(
            protected_available=False,
            protected_summary="Private memory unavailable for this interlocutor.",
        ),
        curiosity=(),
        relationship=RelationshipSnapshot(
            available=False,
            warnings=("private relationship context withheld",),
        ),
        self_model=self_model,
    )


def identity_for_interlocutor(
    identity: IdentityCore,
    interlocutor: InterlocutorContext,
) -> IdentityCore:
    if interlocutor.allows_personal_memory:
        return identity
    relationship = RelationshipModel(
        person_name=interlocutor.display_name or "interlocutor no verificado",
        trust_description="identity is not verified; private relationship context is unavailable",
        familiarity="unverified interaction without access to shared history",
        frequent_topics=(),
        shared_projects=(),
        conversation_style="respectful, natural and privacy-preserving",
        continuity_cues=(),
        evolution=("private relationship state is withheld until identity is verified",),
        shared_history=RelationshipSnapshot(),
    )
    return replace(identity, relationship_model=relationship)


def _observation_summary(interlocutor: InterlocutorContext) -> str:
    if interlocutor.allows_personal_memory:
        return "The current interlocutor may receive owner-scoped conversational context."
    return "The current interlocutor must not receive owner-scoped memory or relationship context."


def _self_without_relationship(self_model, interlocutor: InterlocutorContext):
    mutable_state = dict(self_model.mutable_state)
    identity_core = mutable_state.get("identity_core")
    if isinstance(identity_core, dict):
        safe_identity_core = dict(identity_core)
        safe_identity_core["relationship_model"] = {
            "person_name": interlocutor.display_name or "interlocutor no verificado",
            "trust_description": "private relationship context is unavailable",
            "familiarity": "unverified interaction without shared history",
            "frequent_topics": [],
            "shared_projects": [],
            "conversation_style": "respectful and privacy-preserving",
            "last_interaction_at": "",
            "continuity_cues": [],
            "evolution": [],
            "shared_history": RelationshipSnapshot(
                available=False,
                warnings=("private relationship context withheld",),
            ).to_dict(),
        }
        mutable_state["identity_core"] = safe_identity_core
    return replace(self_model, mutable_state=mutable_state)
