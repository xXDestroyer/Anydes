# Architecture

## Real module boundaries

Python packages retain their original `programa_principal/anydes` and
`anydes_tools` layout so relative imports and ownership remain recognizable.
The accepted candidate overlay remains separately identifiable under
`experimentos/anydes_professional_shell_h1_4/h1_4b_13`; it is not silently mixed
into older production source. Its bootstrap selects reviewed overrides.

## Observation and context

`anydes.cognition.contracts.Observation` represents incoming evidence and origin.
Text, microphone/STT, vision, body, environment and tool results are observations,
not execution authority. `ContextBuilder` composes `ContextProviders` into
`InternalContext` using memory, SELF, goals, emotion, uncertainty, curiosity,
vision, environment, relationship and body providers.

`MemoryItem` records a typed memory tier/category. `MemoryTier` names working,
short, long and protected tiers. Providers adapt existing stores; controlled
writers decide what may persist. Protected memory content is never a generic
public context dump. This release includes contracts, providers and writer code,
not any actual store or encrypted personal payload. Legacy `memoria` source
availability is described in PORTABILITY.md rather than replaced with fake stores.

SELF uses `SelfModel`, `SelfPreferenceStore`, identity/relationship contracts and
controlled self writers. `Goal`/`GoalSnapshot` carry durable/temporary goals and
status. Emotion, uncertainty and curiosity providers contribute bounded context;
their existence does not establish human emotions or a completed initiative system.

## Reasoning, intention and reflection

The cognition package separates reasoning, intention proposals, grounded evidence,
reflection and action management. `Intention`, `ReflectionResult`, `ActionRequest`
and `ActionResult` are real cognitive contracts in cognition/contracts.py.
Reflection may approve, revise, ask confirmation, block or defer. It does not
grant arbitrary filesystem or execution permissions.

## Interaction and task continuity

The tools conversation package resolves an existing public discriminated semantic
contract into `InteractionResolution`. The classifier wire boundary extracts one
unambiguous JSON object; schema and semantic validation remain authoritative.
The resolver permits one repair, then fails closed if unavailable. No local
PendingTask-to-CONTINUE_TASK heuristic replaces semantic classification.

`TaskContract` underlies the pending task's structured semantic state. Required,
optional and defaultable requirements, provenance, unresolved clarification targets
and readiness are kept independently of transcript wording. Runtime owns canonical
task IDs; model updates may reference only the existing task. `PendingTaskState`
and the controller coordinate this state without creating a separate core.

## Actions and artifacts

Planner proposals are lowered to an executable plan using registered capabilities.
PermissionBridge and ToolExecutor check declared contracts, effects, paths and
authorization before worker dispatch. `ToolRequest`/`ToolExecutionResult` are the
tool-layer contracts; they are distinct from cognitive ActionRequest/ActionResult.

`ArtifactRef` identifies an artifact and its state/version/evidence. Dataflow binds
the intended artifact to later operations. Creation and commit are separated;
failed/inconclusive verification must not publish false success. Metadata queries
read the same authoritative reference without creating or running another action.

## Verification and publication

`TestSpec`, `VerificationResult`, `OracleRegistry` and `OracleSelector` separate
execution evidence from independently justified correctness. `units.affine` and
`text.character_count` are host-owned deterministic reference families. Generated
expected hints do not become authoritative just by agreeing with generated code.

The publication layer consumes `RuntimeOutcome` and evidence-backed allowed claims.
Nonempty approved text is shared with chat/voice. The technical-literal adapter
preserves visible approved paths/IDs and scopes mathematical speech preparation.

## Voice and Presence

Parakeet input, hybrid inference and B4 output feed a single shared runtime.
Presence is an observer: `PresenceStateAdapter` maps real activity to IDLE,
LISTENING, THINKING, SPEAKING and WORKING. Local loopback SSE carries small state
payloads, not prompts or memories. AudioTap and speech motion observe real playback
features without owning voice synthesis. Shell renderer selection changes only
representation; it must not duplicate memory, tools or B4.

## Controlled persistence and learning

MemoryWriter, self/relationship writers and curiosity writers are controlled
persistence boundaries. Planned Forge capability extension is a future composition
of generation, sandbox tests, independent verification and authorization, not
unrestricted source self-modification.
