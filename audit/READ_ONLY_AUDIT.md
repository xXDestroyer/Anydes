# Read-only installation audit (before source copying)

Only this separate release directory is writable output. The installation was
inventoried without importing project modules or launching Anydes.

| Candidate area | Files | Bytes |
| --- | ---: | ---: |
| programa_principal/anydes | 336 | 4,778,377 |
| anydes_tools | 320 | 2,877,115 |
| Official capability manifests | 4 | 50,528 |
| Accepted classifier/interaction candidate | 54 | 814,540 |
| Presence source, pruned | 33 | 555,414 |

These are broad candidates, not a copy list. The final allowlist excludes user
configuration, avatar export metadata, empty legacy modules, third-party Python
dependency distributions, runtime/test outputs and diagnostic UI automation.

## Source mapping

- `anydes/cognition/contracts.py`: Observation, MemoryItem, Goal, InternalContext,
  Intention, ReflectionResult and action/result contracts.
- `cognition/context_builder.py`, `providers/`: memory, SELF, goals, emotion,
  uncertainty, curiosity, environment, vision, relationship and body context.
- `cognition/reasoning.py`, `reflection.py`, `action_manager.py` and writer modules:
  inference, reflection, action authorization and controlled persistence.
- `identity/` and cognitive providers: memory/SELF implementation and interfaces,
  not their persisted personal contents. Some old memoria/cerebro files are empty
  compatibility remnants and will not be presented as substantive implementations.
- `anydes_tools/conversation`: interaction classification, task continuity,
  ArtifactRef/dataflow, code generation, verification/oracles and publication.
- Accepted candidate overlay: latest validated versions of the above boundaries.
- `anydes_tools/windows_sandbox.py` plus accepted staging overlay: Windows B2,
  permissions, path boundaries and the validated disappearing-file fix.
- Professional Shell and Presence Lab source: embedded renderer, local state
  adapter, motion and voice feature integration, without session payloads.
- `stt/`, `voz/`, `voice_input/`, `vision/`, `hybrid/`, `realtime/`: adapters only.

## Privacy and security decisions

Real .env is excluded entirely. Its values will never be printed or copied.
Broad candidate scanning found machine-profile paths and two email-like literals
inside vendored dependency tests; the vendored dependency tree will be omitted.
Root .env findings are classified separately with location/category only.

Actual memory stores, profiles/prompts, SELF state, histories, sessions, telemetry,
screenshots, microphones, voice examples, generated user artifacts and databases
are excluded. Source fixtures will be reviewed/sanitized as examples, not confused
with user memory. No existing screenshot is needed; no new screenshot is taken.

No model, virtual environment, binary runtime, cache, rollback or node_modules
will be included. Three.js remains an exact npm dependency, not a copied cache.
Required small bundled JS notices (e.g. Lucide/MIT) will accompany their code.

No project-root LICENSE was found. A licensing note will request an explicit
owner decision; no MIT/GPL/etc. license will be selected automatically.

## Packaging decision

Preserve Python package/import boundaries and the meaningful accepted overlay
layout instead of inventing a mock src hierarchy. Add judge-facing documentation,
sanitized examples, dependency groups and a safe source-validation test runner.
The release is source/documentation, not a claim of one-command full model setup.

No file in the installation has been written, moved, renamed or deleted.
