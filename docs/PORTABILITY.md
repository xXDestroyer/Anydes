# Source-Only Portability and Known Limits

- Original package boundaries are preserved. The accepted overlay is explicit,
  not a new mock implementation or an undocumented merge into production.
- Private .env, user/SELF profiles, memory stores, runtime data and voice references
  are absent. Create your own configuration; do not point this release at the
  author's installation or personal data directories.
- Original startup/integrity code references machine installation profiles and
  manifests that are intentionally not distributed. An anonymized export cannot
  satisfy the old byte-for-byte installation manifest. Do not bypass its checks
  and call that a validated production launch.
- Some legacy modules under `anydes/memoria` and `anydes/cerebro` are empty in the
  audited source tree. Their real contracts/providers/writers are present elsewhere,
  but this export does not reconstruct unavailable MemoryManager/legacy engine
  definitions from bytecode or fabricate replacements. Full runtime imports may
  require maintained source components not available in this audited tree.
- Machine-specific source literals were anonymized in the COPY. Generic DemoUser
  paths inside tests are illustrative technical literals, not deployment paths.
- Avatar code remains, but personal/original avatar textures/models and selector
  preview screenshots are not distributed. Missing previews are documented, not
  replaced with a new identity. The procedural hologram source is included.
- Python vendored dependency distributions were omitted; install JSON Schema and
  other packages from requirements. Three.js is an exact npm dependency; no
  node_modules directory is committed.
- Full historical regression includes Windows/B2/compiler/model/integration tests
  and older diagnostic evidence references. Some need external setup or absent
  historical fixtures. The safe source-release checks do not claim 529 fresh passes.
- B2 requires Windows, the supported compiler layout for C++, and real isolation.
  It must not be made to work by disabling security.

`validate_release.bat` performs source preflight only. It deliberately does not
promise a model-ready UI. Review the preserved `launch_staging.py` and dependency
chain as integration source once these prerequisites have been supplied and audited.
