from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Protocol
from .verification_types import TestSpec, build_runtime_tests
from .verification_types import VerificationRequirement, code_verification_requirement
from .operation_semantics import canonical_descriptor
from .source_names import canonical_source_name


_CONTROL_MARKUP = re.compile(
    r"<tool_call\b|<action_request\b|\[(?:official\.)?(?:developer|filesystem)\."
    r"|\"tool_id\"\s*:|\"capability\"\s*:\s*\"developer\.",
    re.IGNORECASE,
)


CodeTestCase = TestSpec  # Deprecated name retained for existing consumers.


@dataclass(frozen=True)
class CodeGenerationSpec:
    language: str
    goal: str
    functional_requirements: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    constraints: tuple[str, ...]
    dependencies: tuple[str, ...]
    target_filename: str
    tests: tuple[CodeTestCase, ...]
    execution_required: bool
    user_interactive: bool = False
    input_prompts: tuple[str, ...] = ()
    output_clarity: str = "clear"
    basic_error_handling: bool = True
    usability_level: str = "simple"
    comments_when_helpful: bool = True
    verification_requirement: str | None = None
    operation_descriptor: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CodeGenerationSpec":
        language = str(value.get("language", "python")).strip().casefold()
        filename = canonical_source_name(value.get("target_filename", "programa.py"), 'py' if language == 'python' else 'cpp')
        return cls(
            language=language,
            goal=str(value.get("goal", "")).strip(),
            functional_requirements=tuple(str(item) for item in (value.get("functional_requirements") or ())),
            inputs=tuple(str(item) for item in (value.get("inputs") or ())),
            outputs=tuple(str(item) for item in (value.get("outputs") or ())),
            constraints=tuple(str(item) for item in (value.get("constraints") or ())),
            dependencies=tuple(str(item) for item in (value.get("dependencies") or ())),
            target_filename=filename,
            tests=(build_runtime_tests(value['test_cases'], semantic_cases=True,
                operation_descriptor=value.get('operation_descriptor')) if 'test_cases' in value else tuple(
                CodeTestCase.from_dict(dict(item))
                for item in (value.get("tests") or ())
                if isinstance(item, dict)
            )),
            execution_required=bool(value.get("execution_required", True)),
            user_interactive=bool(value.get("user_interactive", False)),
            input_prompts=tuple(
                str(item) for item in (value.get("input_prompts") or ())
            ),
            output_clarity=str(value.get("output_clarity", "clear")),
            basic_error_handling=bool(value.get("basic_error_handling", True)),
            usability_level=str(value.get("usability_level", "simple")),
            comments_when_helpful=bool(value.get("comments_when_helpful", True)),
            verification_requirement=(VerificationRequirement(value["verification_requirement"]).value
                if value.get("verification_requirement") is not None else None),
            operation_descriptor=canonical_descriptor(value.get("operation_descriptor", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeneratedCodeArtifact:
    language: str
    filename: str
    code: str
    tests: tuple[CodeTestCase, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GeneratedCodeArtifact":
        return cls(
            language=str(value.get("language", "")).strip().casefold(),
            filename=canonical_source_name(value.get("filename", ""), 'py' if str(value.get('language','')).casefold() == 'python' else 'cpp'),
            code=str(value.get("code", "")),
            tests=tuple(
                CodeTestCase.from_dict(dict(item))
                for item in (value.get("tests") or ())
                if isinstance(item, dict)
            ),
        )


@dataclass(frozen=True)
class CodeModificationSpec:
    target_artifact: str
    language: str
    requested_change: str
    preserve_requirements: tuple[str, ...]
    new_requirements: tuple[str, ...]
    tests_to_preserve: tuple[CodeTestCase, ...]
    new_tests: tuple[CodeTestCase, ...]
    execution_required: bool
    existing_code: str = ""
    operation_descriptor: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CodeModificationSpec":
        return cls(
            target_artifact=str(value.get("target_artifact", "")).strip(),
            language=str(value.get("language", "python")).strip().casefold(),
            requested_change=str(value.get("requested_change", "")).strip(),
            preserve_requirements=tuple(
                str(item) for item in (value.get("preserve_requirements") or ())
            ),
            new_requirements=tuple(
                str(item) for item in (value.get("new_requirements") or ())
            ),
            tests_to_preserve=tuple(
                CodeTestCase.from_dict(dict(item))
                for item in (value.get("tests_to_preserve") or ())
                if isinstance(item, dict)
            ),
            new_tests=tuple(
                CodeTestCase.from_dict(dict(item))
                for item in (value.get("new_tests") or ())
                if isinstance(item, dict)
            ),
            execution_required=bool(value.get("execution_required", True)),
            existing_code=str(value.get("existing_code", "")),
            operation_descriptor=canonical_descriptor(value.get('operation_descriptor', {})),
        )

    def with_existing_code(self, source: str) -> "CodeModificationSpec":
        from dataclasses import replace

        return replace(self, existing_code=str(source))

    def to_dict(self, *, include_source: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if not include_source:
            payload.pop("existing_code", None)
        return payload


class CodeModifier(Protocol):
    def __call__(self, specification: CodeModificationSpec) -> GeneratedCodeArtifact:
        ...


@dataclass(frozen=True)
class CodeValidationResult:
    valid: bool
    errors: tuple[str, ...]


class CodeGenerator(Protocol):
    def __call__(self, specification: CodeGenerationSpec) -> GeneratedCodeArtifact:
        ...


class CodeValidator:
    @staticmethod
    def validate(
        artifact: GeneratedCodeArtifact,
        specification: CodeGenerationSpec,
    ) -> CodeValidationResult:
        errors: list[str] = []
        language = artifact.language or specification.language
        expected_suffix = ".py" if language == "python" else ".cpp" if language in {"cpp", "c++"} else ""
        if not artifact.code.strip():
            errors.append("generated_code_empty")
        if len(artifact.code.encode("utf-8")) > 500_000:
            errors.append("generated_code_too_large")
        if _CONTROL_MARKUP.search(artifact.code):
            errors.append("generated_code_contains_control_markup")
        if not expected_suffix:
            errors.append("generated_code_language_unsupported")
        if expected_suffix and Path(artifact.filename).suffix.casefold() != expected_suffix:
            errors.append("generated_code_extension_mismatch")
        if artifact.filename != specification.target_filename:
            errors.append("generated_code_filename_mismatch")
        if specification.dependencies:
            errors.append("generated_code_dependencies_not_allowed")
        if language == "python" and artifact.code.strip():
            try:
                ast.parse(artifact.code)
            except SyntaxError:
                errors.append("generated_code_syntax_error")
        if len(artifact.tests) > 8:
            errors.append("generated_code_test_limit_exceeded")
        if len(specification.tests) > 8:
            errors.append("code_spec_test_limit_exceeded")
        requirement = code_verification_requirement(specification)
        if requirement in {VerificationRequirement.EXECUTION,VerificationRequirement.TEST_SUITE} and not specification.execution_required:
            errors.append("code_verification_requires_execution")
        if specification.execution_required and requirement in {VerificationRequirement.NONE,VerificationRequirement.BASIC}:
            errors.append("code_execution_requirement_contradiction")
        if requirement == VerificationRequirement.TEST_SUITE and not (specification.tests or artifact.tests):
            errors.append("code_execution_requires_tests")
        if any(not test.test_id for test in artifact.tests):
            errors.append("generated_code_test_id_required")
        return CodeValidationResult(not errors, tuple(dict.fromkeys(errors)))


def verify_code_test(stdout: str, test: CodeTestCase) -> bool:
    from .typed_verification import VerificationDispatcher
    from .verification_types import VerificationStatus
    return VerificationDispatcher.verify("CODE", test=test, stdout=stdout, exit_code=0,
        trusted_oracle=True).status == VerificationStatus.PASS
