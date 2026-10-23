# Release Validation

No live assistant, model call, microphone, TTS playback, external execution or
application UI was started to build this source release. The accepted 529-test
regression belongs to the original installation and was not rerun here.

## Checks performed on exported code

- Python AST syntax and JSON parsing: see `audit/source_validation_report.md`.
- Portable serialization tests: 10 PASS, 0 FAIL (`tests/test_release_boundaries.py`).
- Existing classifier, code-output and technical-literal suites: 52 PASS, 0 FAIL.
- Existing JavaScript motion, cognitive-motion, runtime-bridge and speech-motion
  tests: 18 PASS, 0 FAIL. Runtime-bridge tests use a fake event source, not a port.

The focused Python suites loaded all Anydes application modules from this release.
The validation interpreter reused the installation's third-party JSON Schema
dependencies read-only because neither available interpreter had that dependency
installed globally. No vendored Python distribution was exported. Install
`requirements.txt` in a new environment to supply it normally.

## Reproduce safe checks

```text
python -B tools/check_source.py
python -B -m unittest discover -s tests -v
python -B tools/run_focused_tests.py
```

For existing browser-motion logic (Node.js, no npm install needed for these tests):

```text
cd experimentos/anydes_presence_lab_v001
node --test tests/*.test.mjs
```

The source includes additional genuine tests for contracts, permissions,
ArtifactRef/dataflow, publication, B2 and oracles. Do not blanket-run all historical
test or acceptance scripts: integration suites may require Windows isolation,
compilers, models, owner-supplied profiles or omitted historical evidence. Some
legacy reporting helpers are retained because tests import them. Runtime collectors
and live replay drivers are excluded from the review package.

Passing static/source-release checks does not assert that a clean machine can
launch the full application. See `PORTABILITY.md` for missing legacy definitions,
nonredistributed assets and runtime prerequisites.
