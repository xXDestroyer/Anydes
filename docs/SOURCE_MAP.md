# Source Map for Judges

| Concern | Real source |
| --- | --- |
| Observation / typed cognitive state | programa_principal/anydes/cognition/contracts.py |
| Context providers | programa_principal/anydes/cognition/providers/ |
| InternalContext composition | programa_principal/anydes/cognition/context_builder.py |
| Reasoning / reflection | programa_principal/anydes/cognition/reasoning.py, reflection.py |
| Action Manager | programa_principal/anydes/cognition/action_manager.py |
| Memory / SELF controlled writers | cognition/memory_writer.py, self_writer.py, identity/ |
| Latest classifier/task/publication work | experimentos/anydes_professional_shell_h1_4/h1_4b_13/overlay/conversation/ |
| Base tool architecture | anydes_tools/ |
| Permissions / filesystem security | anydes_tools/permissions.py, path_policy.py, executor.py |
| Accepted B2 hardening | h1_4b_13/overlay/windows_sandbox.py and its staging selector |
| ArtifactRef / dataflow | anydes_tools/conversation/dataflow.py and related artifact modules |
| TestSpec / oracles | anydes_tools/conversation/verification_types.py, oracle_registry.py, typed_verification.py |
| Professional Shell | experimentos/anydes_professional_shell_h1_4/shell.py, policy.py, assets.py |
| Presence state / speech motion | experimentos/anydes_presence_lab_v001/h1_2/, h1_3/ |
| Actual hologram | experimentos/anydes_presence_lab_v001/app.js, motion.mjs, speech-motion.mjs |
| Hybrid / realtime inference | programa_principal/anydes/hybrid/, realtime/, continuity_v2/ |
| Voice / STT / vision adapters | programa_principal/anydes/voz/, voice_input/, stt/, vision/ |

Paths preserve the actual import architecture; the table does not imply that all
original machine-specific entry points are directly runnable from the export.
