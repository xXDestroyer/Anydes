# Source Validation

- Python syntax: 520 files checked; 0 failures.
- JSON parsing: 18 files checked.
- No application startup, model calls, bytecode writes or ports opened.
- Real classifier/code-output/literal suites: 52 PASS, 0 FAIL (see focused_tests.json).
- Portable serialization tests: 10 PASS, 0 FAIL.
- Existing JavaScript motion suites: 18 PASS, 0 FAIL.
- JavaScript syntax: 15 files checked with Node --check; 0 failures.
- Import sanity: no Anydes application modules loaded from outside this release.
- JSON Schema was supplied read-only from existing third-party dependencies for tests.
- Full application imports remain limited by dependencies and legacy source gaps; see docs/PORTABILITY.md.
- The historical 529-test installation regression is not a fresh release test count.
