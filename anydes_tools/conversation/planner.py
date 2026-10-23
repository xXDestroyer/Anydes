from __future__ import annotations

from pathlib import Path
import re
import unicodedata
from uuid import uuid4

from .contracts import (
    ConversationToolState,
    PlannedToolAction,
    StructuredActionPlan,
    UserTaskSpec,
)
from .document_transform import (
    DocumentOperation,
    TargetResolver,
    parse_semantic_transformation,
    resolve_operation_intent,
)
from .document_model import parse_document_edit_spec
from .intent_normalization import (
    asks_for_full_path,
    extract_explicit_filename_span,
    normalize_control_text,
    resolve_document_format,
    sanitize_windows_stem,
)
from .content_generation import ContentGenerationSpec, parse_content_generation_spec
from .delete_plan import parse_delete_selection
from .document_conversion import (
    DocumentConversionSpec,
    SourceResolver,
    extract_source_stem,
)
from .structured_text import ContentBuilder, TextCreationSpec, parse_creation_spec, parse_transform_spec


_WINDOWS_PATH = re.compile(
    r"(?P<path>[A-Za-z]:\\(?:[^\r\n\"']+|\"[^\"]+\")+)"
)
_QUOTED = re.compile(r"[\"']([^\"']+)[\"']")

_DOCUMENT_FORMATS = {
    "txt": {
        "extension": ".txt",
        "capability": "document.create_txt",
        "aliases": (r"\.txt\b", r"\btxt\b", r"\barchivo\s+de\s+texto\b", r"\btexto\b"),
    },
    "markdown": {
        "extension": ".md",
        "capability": "document.create_markdown",
        "aliases": (r"\.md\b", r"\bmarkdown\b"),
    },
    "docx": {
        "extension": ".docx",
        "capability": "document.create_docx",
        "aliases": (r"\.docx\b", r"\bdocx\b", r"\bword\b"),
    },
    "pdf": {
        "extension": ".pdf",
        "capability": "document.create_pdf",
        "aliases": (r"\.pdf\b", r"\bpdf\b"),
    },
}
_FORMAT_ALIAS_PRIORITY = ("docx", "pdf", "markdown", "txt")


def plan_from_text(
    text: str,
    state: ConversationToolState,
    *,
    content_builder: ContentBuilder | None = None,
    requested_operations: tuple[str, ...] = (),
) -> PlannedToolAction | None:
    raw = text.strip()
    control_text = normalize_control_text(raw)
    normalized = _normalize(control_text)
    if not normalized:
        return None

    if _asks_about_pdf(normalized):
        return None
    if _targets_core(normalized):
        return PlannedToolAction(
            intent="forbidden",
            capabilities=("core.modify",),
            target_path=str(Path(state.active_workspace).parents[1] / "programa_principal"),
            source_text=raw,
        )
    if _video_edit(normalized):
        return PlannedToolAction(
            intent="missing",
            capabilities=("media.video.trim",),
            source_text=raw,
        )

    explicit_path = _extract_path(raw)
    explicit_filename = _extract_filename_entity(raw)
    referenced = _referenced_file(normalized, state)
    artifact_target = state.recent_artifact_state.resolve(
        raw,
        explicit_filename=explicit_filename,
        require_unique=True,
    )
    target = TargetResolver.resolve(
        explicit_path,
        state,
        explicit_filename,
    ) if (explicit_path or explicit_filename) else (artifact_target or referenced)

    if asks_for_full_path(raw) and target:
        return PlannedToolAction(
            intent="show_full_path",
            capabilities=("filesystem.inspect",),
            arguments={"path": target},
            target_path=target,
            source_text=raw,
        )

    delete_selection = parse_delete_selection(raw, state.active_workspace)
    if delete_selection is not None:
        return PlannedToolAction(
            intent="delete_collection",
            capabilities=("filesystem.search", "filesystem.delete"),
            operation="delete_collection_request",
            arguments={
                "root": delete_selection.workspace,
                "pattern": delete_selection.pattern,
                "delete_scope": delete_selection.scope.value,
                "recursive": delete_selection.recursive,
            },
            target_path=delete_selection.workspace,
            source_text=raw,
        )

    operation_intent = resolve_operation_intent(raw)
    operation_kind = operation_intent.operation
    if operation_kind == DocumentOperation.RENAME_EXISTING:
        return _rename_action(raw, state, target)
    if operation_kind == DocumentOperation.COPY_EXISTING:
        return _copy_action(raw, state, target)
    if operation_kind == DocumentOperation.CONVERT_EXISTING:
        return _document_conversion_action(
            raw,
            state,
            explicit_path=explicit_path,
            explicit_filename=explicit_filename,
            resolved_source=target,
        )
    if operation_kind != DocumentOperation.CREATE_NEW and target:
        generation = parse_content_generation_spec(
            raw,
            purpose="edit_existing_document",
            default_words=180,
            allow_open_content=False,
        )
        if generation is not None:
            return _generated_edit_action(raw, target, generation)
    if operation_kind in {
        DocumentOperation.EDIT_EXISTING,
        DocumentOperation.TRANSFORM_EXISTING,
    }:
        target = target or TargetResolver.resolve(explicit_path, state, explicit_filename)
        structural_edit = parse_document_edit_spec(raw, target)
        if structural_edit is not None:
            return PlannedToolAction(
                intent="structural_document_edit",
                capabilities=("document.read_document", "document.edit_document"),
                operation="structural_document_edit",
                arguments={"path": target, "edit_spec": structural_edit.to_dict()},
                target_path=target,
                source_text=raw,
            )
        if _append_text(normalized):
            addition = _after_marker(raw, ("agrega", "añade", "anade"))
            return PlannedToolAction(
                intent="append_document",
                capabilities=("document.read_document", "document.edit_document"),
                operation="append_document",
                arguments={"path": target, "addition": addition},
                target_path=target,
                source_text=raw,
            )
        transform = parse_transform_spec(raw)
        if transform is not None:
            return PlannedToolAction(
                intent="structured_edit_text",
                capabilities=("document.read_document", "document.edit_document"),
                operation="structured_edit_text",
                arguments={"path": target, "transform": transform.payload()},
                target_path=target,
                source_text=raw,
            )
        semantic = parse_semantic_transformation(raw, target)
        if semantic is not None:
            return _semantic_transform_action(raw, target, semantic.to_dict())
        return PlannedToolAction(
            intent="invalid_document_transform",
            capabilities=("document.read_document", "document.edit_document"),
            operation="invalid_document_transform",
            arguments={"path": target},
            target_path=target,
            source_text=raw,
        )

    # A creation request owns its embedded layout/annotation clauses. They are
    # part of UserTaskSpec, not a follow-up edit against a nonexistent file.
    created = _create_document(raw, normalized, state, content_builder=content_builder)
    if created is not None:
        return created

    transform = parse_transform_spec(raw)
    if transform is not None:
        if not target and state.active_document_ref is not None:
            target = state.active_document_ref.absolute_canonical_path
        elif not target:
            target = state.active_document
    if transform is not None:
        return PlannedToolAction(
            intent="structured_edit_text",
            capabilities=("document.read_document", "document.edit_document"),
            operation="structured_edit_text",
            arguments={"path": target, "transform": transform.payload()},
            target_path=target,
            source_text=raw,
        )

    if _error_inquiry(normalized) and state.last_error:
        return PlannedToolAction(
            intent="inspect_error",
            capabilities=("developer.inspect_error",),
            arguments={"stderr": str(state.last_error.get("stderr", ""))},
            source_text=raw,
        )

    if _join_texts(normalized):
        root = explicit_path or state.active_workspace
        return PlannedToolAction(
            intent="text_files_to_pdf",
            capabilities=("filesystem.search", "filesystem.read", "document.create_pdf"),
            operation="text_files_to_pdf",
            arguments={
                "root": root,
                "pattern": "*.txt",
                "destination": str(Path(root) / "textos_unidos.pdf"),
            },
            target_path=root,
            source_text=raw,
        )

    if requested_operations == ("CREATE_ARCHIVE",) or _compress(normalized):
        selected_ref = state.recent_artifact_state.resolve_ref(raw, require_unique=True,
            excluded_names=(_explicit_requested_stem(raw),)) if not explicit_path else None
        source = (explicit_path if explicit_path and Path(explicit_path).is_absolute() else
            selected_ref.path if selected_ref else state.recent_artifact_state.resolve(raw, require_unique=True,
                excluded_names=(_explicit_requested_stem(raw),)))
        if not source:
            return PlannedToolAction(intent="needs_artifact_selection", capabilities=(),
                source_text=raw, explicit_request=True)
        source_path = Path(source)
        destination = Path(state.active_workspace) / _requested_filename(raw, source_path.stem, ".zip")
        return PlannedToolAction(
            intent="compress_zip",
            capabilities=("archive.compress_zip",),
            arguments={"sources": [str(source_path)], "destination": str(destination)},
            target_path=str(source_path),
            source_text=raw,
            resolved_artifacts=tuple(ref for ref in (selected_ref or state.recent_artifact_state.ref_by_path(str(source_path)),) if ref),
        )

    if _delete(normalized):
        return PlannedToolAction(
            intent="delete",
            capabilities=("filesystem.delete",),
            arguments={"path": target or ""},
            target_path=target or "",
            source_text=raw,
        )

    if _run_python(normalized):
        code = _code_payload(raw)
        if target and Path(target).suffix.casefold() == ".py":
            return PlannedToolAction(
                intent="run_python_file",
                capabilities=("developer.run_python",),
                arguments={"path": target},
                target_path=target,
                source_text=raw,
            )
        return PlannedToolAction(
            intent="run_inline_python",
            capabilities=("developer.write_code", "developer.run_python"),
            operation="run_inline_python",
            arguments={"code": code or "print('Hola desde Anydes')"},
            source_text=raw,
        )

    if _patch_code(normalized):
        old, new = _replacement(raw)
        if not target or not old:
            return PlannedToolAction(
                intent="needs_patch_details",
                capabilities=("developer.patch_code",),
                target_path=target or "",
                source_text=raw,
            )
        return PlannedToolAction(
            intent="patch_code",
            capabilities=("developer.patch_code",),
            arguments={"path": target, "old_text": old, "new_text": new},
            target_path=target,
            source_text=raw,
        )

    if _append_text(normalized):
        addition = _after_marker(raw, ("agrega", "añade", "anade"))
        return PlannedToolAction(
            intent="append_document",
            capabilities=("document.read_document", "document.edit_document"),
            operation="append_document",
            arguments={"path": target or "", "addition": addition},
            target_path=target or "",
            source_text=raw,
        )

    if _read(normalized):
        return PlannedToolAction(
            intent="read_document",
            capabilities=("document.read_document",),
            arguments={"path": target or ""},
            target_path=target or "",
            source_text=raw,
        )

    return None


def _create_document(
    raw: str,
    normalized: str,
    state: ConversationToolState,
    *,
    content_builder: ContentBuilder | None = None,
) -> PlannedToolAction | None:
    if not _creation_requested(normalized):
        return None
    format_id = _requested_document_format(normalized)
    if format_id is None:
        return None
    descriptor = _DOCUMENT_FORMATS[format_id]
    suffix = str(descriptor["extension"])
    capability = str(descriptor["capability"])
    explicit = _extract_path(raw)
    literal_content = _content_payload(raw)
    try:
        creation_spec = parse_creation_spec(
            raw,
            literal_content,
            builder=content_builder,
        )
    except (TypeError, ValueError, RuntimeError):
        creation_spec = None
    deterministic_content = (
        creation_spec is not None
        and (creation_spec.source_kind != "literal" or bool(creation_spec.render()))
    )
    generation = None if deterministic_content else parse_content_generation_spec(
        raw,
        purpose="create_document",
    )
    if generation is not None:
        stem = _safe_filename_stem(generation.content_kind)
        requested_name = _requested_filename(raw, stem, suffix)
        path = (
            Path(explicit)
            if explicit
            else _next_available(Path(state.active_workspace) / requested_name)
        )
        if not path.is_absolute():
            path = Path(state.active_workspace) / path
        if path.suffix.casefold() != suffix:
            path = path.with_suffix(suffix)
        return _generated_create_action(
            raw,
            state,
            format_id,
            capability,
            path,
            generation,
            document_title=(
                path.stem if _explicit_requested_stem(raw) else generation.topic
            ),
        )
    if creation_spec is None or not creation_spec.render().strip():
        return None
    stem = _safe_filename_stem(creation_spec.suggested_stem)
    requested_name = _requested_filename(raw, stem, suffix)
    path = (
        Path(explicit)
        if explicit
        else _next_available(Path(state.active_workspace) / requested_name)
    )
    if path.suffix.casefold() != suffix:
        path = path.with_suffix(suffix)
    content = creation_spec.render()
    task_id = f"task:{uuid4().hex}"
    destination = str(path.parent.resolve(strict=False))
    task_spec = UserTaskSpec(
        task_id=task_id,
        goal="create_document",
        target_type="document",
        output_format=format_id,
        filename=path.name,
        content_requirements={"creation_spec": creation_spec.payload()},
        layout_requirements={
            "layout": creation_spec.layout,
            "separator": creation_spec.separator,
            "item_count": len(creation_spec.values),
        },
        constraints=("preserve_requested_format", "verify_after_write"),
        destination=destination,
    )
    arguments = {
        "path": str(path),
        "content": content,
        "overwrite": path.exists(),
        "creation_spec": {
            "source_kind": creation_spec.source_kind,
            "layout": creation_spec.layout,
            "separator": creation_spec.separator,
            "suffix": creation_spec.suffix,
            "item_count": len(creation_spec.values),
            "conditions": [dict(item) for item in creation_spec.conditions],
        },
    }
    action_plan = StructuredActionPlan(
        plan_id=f"plan:{uuid4().hex}",
        action_type="EXACT_TOOL",
        tool_id="official.document",
        capability=capability,
        arguments=arguments,
        workspace=destination,
        task_id=task_id,
    )
    capability_framed = bool(re.search(
        r"\bpuedes\s+(?:crear|hacer|generar|preparar)\b",
        normalized,
    ))
    return PlannedToolAction(
        intent="create_document",
        capabilities=(capability,),
        operation="verified_create_document",
        arguments=arguments,
        target_path=str(path),
        source_text=raw,
        explicit_request=(
            not capability_framed
            or creation_spec.source_kind != "literal"
            or bool(content)
            or bool(explicit)
        ),
        task_spec=task_spec,
        structured_action_plan=action_plan,
    )


def _generated_create_action(
    raw: str,
    state: ConversationToolState,
    format_id: str,
    capability: str,
    path: Path,
    generation: ContentGenerationSpec,
    document_title: str,
) -> PlannedToolAction:
    del state
    task_id = f"task:{uuid4().hex}"
    path = path.resolve(strict=False)
    arguments = {
        "path": str(path),
        "overwrite": path.exists(),
        "content_generation_spec": generation.to_dict(),
        "document_title": document_title,
    }
    task = UserTaskSpec(
        task_id=task_id,
        goal="generate_document",
        target_type="document",
        output_format=format_id,
        filename=path.name,
        content_requirements={"content_generation_spec": generation.to_dict()},
        layout_requirements={"structure": generation.structure},
        constraints=("generate_before_write", "verify_after_write"),
        destination=str(path.parent),
    )
    plan = StructuredActionPlan(
        plan_id=f"plan:{uuid4().hex}",
        action_type="EXACT_TOOL",
        tool_id="official.document",
        capability=capability,
        arguments=arguments,
        workspace=str(path.parent),
        task_id=task_id,
    )
    return PlannedToolAction(
        intent="generate_document",
        capabilities=(capability,),
        operation="generated_create_document",
        arguments=arguments,
        target_path=str(path),
        source_text=raw,
        task_spec=task,
        structured_action_plan=plan,
    )


def _copy_action(
    raw: str,
    state: ConversationToolState,
    target: str,
) -> PlannedToolAction:
    if not target:
        return PlannedToolAction(
            intent="copy_missing_target",
            capabilities=("filesystem.copy",),
            operation="copy_existing",
            source_text=raw,
        )
    source = Path(target).resolve(strict=False)
    requested = extract_explicit_filename_span(raw)
    if requested is not None and _normalize(requested.display_name) != _normalize(source.stem):
        destination_name = requested.display_name
        if not Path(destination_name).suffix:
            destination_name += source.suffix
        destination = source.with_name(destination_name)
    else:
        destination = _next_available(source.with_name(f"{source.stem} copia{source.suffix}"))
    task_id = f"task:{uuid4().hex}"
    arguments = {
        "source": str(source),
        "destination": str(destination.resolve(strict=False)),
        "overwrite": destination.exists(),
    }
    task = UserTaskSpec(
        task_id=task_id,
        goal="copy_existing",
        target_type="artifact",
        output_format=source.suffix.casefold().removeprefix("."),
        filename=destination.name,
        content_requirements={"source_path": str(source)},
        constraints=("preserve_source", "verify_copy"),
        destination=str(destination.parent),
    )
    plan = StructuredActionPlan(
        plan_id=f"plan:{uuid4().hex}",
        action_type="EXACT_TOOL",
        tool_id="official.filesystem",
        capability="filesystem.copy",
        arguments=arguments,
        workspace=str(destination.parent),
        task_id=task_id,
    )
    return PlannedToolAction(
        intent="copy_artifact",
        capabilities=("filesystem.copy",),
        operation="copy_existing",
        arguments=arguments,
        target_path=str(source),
        source_text=raw,
        task_spec=task,
        structured_action_plan=plan,
    )
def _document_conversion_action(
    raw: str,
    state: ConversationToolState,
    *,
    explicit_path: str,
    explicit_filename: str,
    resolved_source: str = "",
) -> PlannedToolAction:
    target_format = _conversion_target_format(raw)
    descriptor = _DOCUMENT_FORMATS.get(target_format)
    if descriptor is None:
        return PlannedToolAction(
            intent="invalid_document_conversion",
            capabilities=("document.read_document",),
            operation="invalid_document_conversion",
            source_text=raw,
        )
    source_filename = explicit_filename
    if source_filename and Path(source_filename).suffix.casefold() == descriptor["extension"]:
        source_filename = ""
    source = resolved_source or SourceResolver.resolve(
        explicit_path=explicit_path,
        explicit_filename=source_filename,
        explicit_stem=extract_source_stem(raw),
        active_document=(
            state.active_document_ref.absolute_canonical_path
            if state.active_document_ref is not None
            else state.active_document
        ),
        recent_files=state.recent_files,
        active_workspace=state.active_workspace,
    )
    if not source:
        return PlannedToolAction(
            intent="invalid_document_conversion",
            capabilities=("document.read_document",),
            operation="invalid_document_conversion",
            source_text=raw,
        )
    source_path = Path(source).resolve(strict=False)
    target_path = source_path.with_suffix(str(descriptor["extension"]))
    specification = DocumentConversionSpec(
        source_path=str(source_path),
        target_path=str(target_path),
        source_format=source_path.suffix.casefold().removeprefix("."),
        target_format=target_format,
    )
    task_id = f"task:{uuid4().hex}"
    arguments = {
        "conversion_spec": specification.to_dict(),
        "source": str(source_path),
        "destination": str(target_path),
        "overwrite": target_path.exists(),
    }
    capability = str(descriptor["capability"])
    task = UserTaskSpec(
        task_id=task_id,
        goal="convert_document",
        target_type="document",
        output_format=target_format,
        filename=target_path.name,
        content_requirements={"source_path": str(source_path)},
        constraints=("preserve_source_content", "atomic_commit", "verify_after_write"),
        destination=str(target_path.parent),
    )
    plan = StructuredActionPlan(
        plan_id=f"plan:{uuid4().hex}",
        action_type="COMPOSABLE",
        tool_id="official.document",
        capability=capability,
        arguments=arguments,
        workspace=str(target_path.parent),
        task_id=task_id,
    )
    return PlannedToolAction(
        intent="convert_document",
        capabilities=("document.read_document", capability, "filesystem.atomic_replace"),
        operation="document_conversion",
        arguments=arguments,
        target_path=str(target_path),
        source_text=raw,
        task_spec=task,
        structured_action_plan=plan,
    )


def _rename_action(
    raw: str,
    state: ConversationToolState,
    target: str,
) -> PlannedToolAction:
    target = target or state.recent_artifact_state.resolve(raw)
    new_name = _extract_rename_destination(raw)
    if target and new_name and not Path(new_name).suffix:
        new_name += Path(target).suffix
    return PlannedToolAction(
        intent="rename" if new_name else "rename_missing_name",
        capabilities=("filesystem.rename",),
        operation="rename_existing",
        arguments={"path": target, "new_name": new_name},
        target_path=target,
        source_text=raw,
    )


def _generated_edit_action(
    raw: str,
    target: str,
    generation: ContentGenerationSpec,
) -> PlannedToolAction:
    path = Path(target).resolve(strict=False)
    task_id = f"task:{uuid4().hex}"
    mode = "REPLACE" if re.search(r"\b(?:reemplaza\s+todo|sobrescrib\w*)\b", _normalize(raw)) else "APPEND"
    arguments = {
        "path": str(path),
        "mode": mode,
        "content_generation_spec": generation.to_dict(),
    }
    task = UserTaskSpec(
        task_id=task_id,
        goal="generate_document_edit",
        target_type="document",
        output_format=path.suffix.casefold().removeprefix(".") or "text",
        filename=path.name,
        content_requirements={"content_generation_spec": generation.to_dict()},
        layout_requirements={"mode": mode},
        constraints=("edit_existing_only", "generate_before_write", "verify_after_write"),
        destination=str(path.parent),
    )
    plan = StructuredActionPlan(
        plan_id=f"plan:{uuid4().hex}",
        action_type="COMPOSABLE",
        tool_id="official.document",
        capability="document.edit_document",
        arguments=arguments,
        workspace=str(path.parent),
        task_id=task_id,
    )
    return PlannedToolAction(
        intent="generate_document_edit",
        capabilities=("document.read_document", "document.edit_document"),
        operation="generated_document_edit",
        arguments=arguments,
        target_path=str(path),
        source_text=raw,
        task_spec=task,
        structured_action_plan=plan,
    )


def _semantic_transform_action(
    raw: str,
    target: str,
    transformation: dict,
) -> PlannedToolAction:
    path = Path(target).resolve(strict=False)
    task_id = f"task:{uuid4().hex}"
    arguments = {"path": str(path), "transformation": transformation}
    task = UserTaskSpec(
        task_id=task_id,
        goal="transform_document",
        target_type="document",
        output_format=path.suffix.casefold().removeprefix(".") or "text",
        filename=path.name,
        content_requirements={"transformation": transformation},
        layout_requirements={"preserve_formatting": True},
        constraints=("edit_existing_only", "verify_after_write"),
        destination=str(path.parent),
    )
    plan = StructuredActionPlan(
        plan_id=f"plan:{uuid4().hex}",
        action_type="COMPOSABLE",
        tool_id="official.document",
        capability="document.edit_document",
        arguments=arguments,
        workspace=str(path.parent),
        task_id=task_id,
    )
    return PlannedToolAction(
        intent="document_transform",
        capabilities=("document.read_document", "document.edit_document"),
        operation="document_transform",
        arguments=arguments,
        target_path=str(path),
        source_text=raw,
        task_spec=task,
        structured_action_plan=plan,
    )


def looks_like_unresolved_tool_request(text: str) -> bool:
    """Recognize an operational request that must not fall through to chat."""

    normalized = _normalize(text)
    action = bool(re.search(
        r"\b(?:cre\w*|hac\w*|hag\w*|haz\w*|gener\w*|prepar\w*|edit\w*|modific\w*|"
        r"guard\w*|escrib\w*|borr\w*|elimin\w*|comprim\w*|ejecut\w*)\b",
        normalized,
    ))
    artifact = bool(re.search(
        r"\b(?:archivo|documento|txt|docx|word|pdf|markdown|zip|codigo|carpeta)\b",
        normalized,
    ))
    return action and artifact


def _creation_requested(normalized: str) -> bool:
    if not re.search(r"\b(?:cre\w*|hac\w*|hag\w*|haz\w*|gener\w*|prepar\w*)\b", normalized):
        return False
    return not bool(re.fullmatch(
        r"(?:anydes\s+)?(?:puedes|podrias)\s+(?:crear|hacer|generar|preparar)\s+"
        r"(?:archivos?|documentos?|txt|word|pdf)\s*[?]?",
        normalized.strip(),
    ))


def validate_structured_action_plan(action: PlannedToolAction, registry) -> tuple[str, ...]:
    plan = action.structured_action_plan
    task = action.task_spec
    if plan is None and task is None:
        return ()
    if plan is None or task is None:
        return ("missing_task_or_action_plan",)
    errors: list[str] = []
    if task.goal == 'retain_unverified':
        target = Path(str(plan.arguments.get('path',''))).resolve(strict=False)
        if action.operation != 'retain_unverified' or action.capabilities != ('developer.write_code',):
            errors.append('quality_retention_contract_mismatch')
        if target.name != task.filename or target.parent != Path(task.destination).resolve(strict=False):
            errors.append('quality_retention_target_mismatch')
        if not registry.find_by_capability('developer.write_code') or not registry.find_by_capability('filesystem.atomic_replace'):
            errors.append('quality_retention_capability_unavailable')
        return tuple(errors)
    if task.goal == "copy_existing":
        source = Path(str(plan.arguments.get("source", ""))).resolve(strict=False)
        target = Path(str(plan.arguments.get("destination", ""))).resolve(strict=False)
        workspace = Path(plan.workspace).resolve(strict=False)
        if plan.capability != "filesystem.copy" or action.capabilities != ("filesystem.copy",):
            errors.append("copy_capability_mismatch")
        if not registry.find_by_capability("filesystem.copy"):
            errors.append("capability_unavailable:filesystem.copy")
        if not source.is_file():
            errors.append("copy_source_missing")
        if source == target:
            errors.append("copy_source_equals_target")
        if target.parent != workspace or target.name != task.filename:
            errors.append("copy_target_mismatch")
        if source.suffix.casefold() != target.suffix.casefold():
            errors.append("copy_format_mismatch")
        return tuple(errors)
    if task.goal == "convert_document":
        conversion = DocumentConversionSpec.from_dict(
            dict(plan.arguments.get("conversion_spec", {}))
        )
        source = Path(conversion.source_path).resolve(strict=False)
        target = Path(conversion.target_path).resolve(strict=False)
        descriptor = _DOCUMENT_FORMATS.get(conversion.target_format)
        if descriptor is None:
            errors.append("conversion_format_unsupported")
        else:
            expected_capability = str(descriptor["capability"])
            if plan.capability != expected_capability:
                errors.append("conversion_capability_mismatch")
            if target.suffix.casefold() != descriptor["extension"]:
                errors.append("conversion_target_extension_mismatch")
            if action.capabilities != (
                "document.read_document",
                expected_capability,
                "filesystem.atomic_replace",
            ):
                errors.append("conversion_composition_mismatch")
            for capability in action.capabilities:
                if not registry.find_by_capability(capability):
                    errors.append(f"capability_unavailable:{capability}")
        if source == target:
            errors.append("conversion_source_equals_target")
        if not source.is_file():
            errors.append("conversion_source_missing")
        if target.name != task.filename or target.parent != Path(plan.workspace).resolve(strict=False):
            errors.append("conversion_target_mismatch")
        return tuple(errors)
    if task.goal in {"generate_document", "generate_document_edit"}:
        target = Path(str(plan.arguments.get("path", ""))).resolve(strict=False)
        workspace = Path(plan.workspace).resolve(strict=False)
        generation = dict(plan.arguments.get("content_generation_spec", {}))
        if not generation or not str(generation.get("topic", "")).strip():
            errors.append("content_generation_spec_missing")
        if target.name != task.filename or target.parent != workspace:
            errors.append("generated_document_target_mismatch")
        if task.goal == "generate_document_edit":
            if plan.capability != "document.edit_document":
                errors.append("generated_edit_capability_mismatch")
            if action.capabilities != ("document.read_document", "document.edit_document"):
                errors.append("generated_edit_composition_mismatch")
            if not target.is_file():
                errors.append("generated_edit_target_missing")
            if target.suffix.casefold() not in {".txt", ".md", ".docx"}:
                errors.append("generated_edit_format_unsupported")
        else:
            descriptor = _DOCUMENT_FORMATS.get(task.output_format)
            if descriptor is None:
                errors.append("unsupported_output_format")
            else:
                if plan.capability != descriptor["capability"]:
                    errors.append("generated_create_capability_mismatch")
                if target.suffix.casefold() != descriptor["extension"]:
                    errors.append("generated_create_extension_mismatch")
        if not registry.find_by_capability(plan.capability):
            errors.append("capability_unavailable")
        return tuple(errors)
    if task.goal == "transform_document":
        target = Path(str(plan.arguments.get("path", ""))).resolve(strict=False)
        workspace = Path(plan.workspace).resolve(strict=False)
        transformation = dict(plan.arguments.get("transformation", {}))
        transformed_target = Path(
            str(transformation.get("target_path", ""))
        ).resolve(strict=False)
        if plan.capability != "document.edit_document":
            errors.append("capability_transform_mismatch")
        if action.capabilities != ("document.read_document", "document.edit_document"):
            errors.append("transform_composition_mismatch")
        if target != transformed_target:
            errors.append("transform_target_mismatch")
        if target.name != task.filename or target.parent != workspace:
            errors.append("transform_workspace_mismatch")
        if not target.is_file():
            errors.append("transform_target_missing")
        if transformation.get("operation") not in {"APPEND_TO_MATCH"}:
            errors.append("transform_operation_invalid")
        if not str(transformation.get("payload", "")).strip():
            errors.append("transform_payload_missing")
        condition = dict(transformation.get("condition", {}))
        if condition.get("kind") != "SEMANTIC" or not str(condition.get("description", "")).strip():
            errors.append("transform_condition_invalid")
        if not registry.find_by_capability(plan.capability):
            errors.append("capability_unavailable")
        return tuple(errors)
    if task.goal == "generate_code":
        target = Path(str(plan.arguments.get("path", ""))).resolve(strict=False)
        workspace = Path(plan.workspace).resolve(strict=False)
        code_spec = dict(plan.arguments.get("code_generation_spec", {}))
        language = str(code_spec.get("language", "")).casefold()
        expected_suffix = ".py" if language == "python" else ".cpp" if language in {"cpp", "c++"} else ""
        if not code_spec or not str(code_spec.get("goal", "")).strip():
            errors.append("code_generation_spec_missing")
        if not expected_suffix:
            errors.append("code_language_unsupported")
        if target.suffix.casefold() != expected_suffix:
            errors.append("code_extension_mismatch")
        if target.name != task.filename or target.parent != workspace:
            errors.append("code_target_mismatch")
        if plan.capability != "developer.write_code":
            errors.append("code_primary_capability_mismatch")
        if "developer.write_code" not in action.capabilities:
            errors.append("code_write_capability_missing")
        for capability in action.capabilities:
            if not registry.find_by_capability(capability):
                errors.append(f"capability_unavailable:{capability}")
        return tuple(errors)
    if task.goal == "modify_code":
        target = Path(str(plan.arguments.get("path", ""))).resolve(strict=False)
        workspace = Path(plan.workspace).resolve(strict=False)
        modification = dict(plan.arguments.get("code_modification_spec", {}))
        if not target.is_file():
            errors.append("code_modification_target_missing")
        if target.name != task.filename or target.parent != workspace:
            errors.append("code_modification_target_mismatch")
        if plan.capability != "developer.write_code":
            errors.append("code_modification_primary_capability_mismatch")
        if not str(modification.get("requested_change", "")).strip():
            errors.append("code_modification_change_missing")
        required = {"developer.read_code", "developer.write_code"}
        if not required.issubset(set(action.capabilities)):
            errors.append("code_modification_capabilities_missing")
        for capability in action.capabilities:
            if not registry.find_by_capability(capability):
                errors.append(f"capability_unavailable:{capability}")
        return tuple(errors)
    if task.goal == "general_tool_composition":
        general = dict(plan.arguments.get("general_task_plan", {}))
        steps = tuple(general.get("steps", ()))
        if not steps:
            errors.append("general_composition_steps_missing")
        if len(steps) > 8:
            errors.append("general_composition_step_limit")
        declared = tuple(str(item) for item in general.get("required_capabilities", ()))
        if declared != action.capabilities:
            errors.append("general_composition_capability_mismatch")
        if plan.capability != (declared[0] if declared else ""):
            errors.append("general_composition_primary_capability_mismatch")
        for capability in declared:
            if not registry.find_by_capability(capability):
                errors.append(f"capability_unavailable:{capability}")
        return tuple(errors)
    descriptor = _DOCUMENT_FORMATS.get(task.output_format)
    if descriptor is None:
        errors.append("unsupported_output_format")
        return tuple(errors)
    expected_capability = str(descriptor["capability"])
    expected_extension = str(descriptor["extension"])
    target = Path(str(plan.arguments.get("path", ""))).resolve(strict=False)
    workspace = Path(plan.workspace).resolve(strict=False)
    if plan.capability != expected_capability or action.capabilities != (expected_capability,):
        errors.append("capability_format_mismatch")
    if target.suffix.casefold() != expected_extension:
        errors.append("extension_format_mismatch")
    if target.name != task.filename:
        errors.append("filename_mismatch")
    if target.parent != workspace or workspace != Path(task.destination).resolve(strict=False):
        errors.append("workspace_mismatch")
    if not registry.find_by_capability(plan.capability):
        errors.append("capability_unavailable")
    try:
        expected = TextCreationSpec.from_payload(
            dict(task.content_requirements.get("creation_spec", {}))
        ).render()
    except (TypeError, ValueError):
        errors.append("invalid_content_specification")
    else:
        if str(plan.arguments.get("content", "")) != expected:
            errors.append("content_plan_mismatch")
    return tuple(errors)


def _requested_document_format(normalized: str) -> str | None:
    resolved = resolve_document_format(normalized)
    if resolved is not None:
        return "markdown" if resolved.value == "markdown" else resolved.value
    explicit = re.search(r"\.(txt|md|docx|pdf)\b", normalized)
    if explicit:
        return "markdown" if explicit.group(1) == "md" else explicit.group(1)
    for format_id in _FORMAT_ALIAS_PRIORITY:
        descriptor = _DOCUMENT_FORMATS[format_id]
        if any(re.search(pattern, normalized) for pattern in descriptor["aliases"]):
            return format_id
    if re.search(r"\b(?:archivo|documento)\b", normalized):
        return "txt"
    return None


def _safe_filename_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("._-")
    return cleaned or "nota"


def _requested_filename(text: str, suggested_stem: str, suffix: str) -> str:
    named = _explicit_requested_stem(text)
    if named:
        if Path(named).suffix.casefold() == suffix.casefold():
            return named
        return f"{named}{suffix}"
    explicit = _direct_filename_entity(text)
    if explicit:
        stem = re.sub(r"[<>:\"/\\|?*]+", "_", Path(explicit).stem).strip(" .")
        return f"{stem}{suffix}"
    version = re.search(r"\bversion\s+(\d+)\b", _normalize(text))
    if version:
        return f"{suggested_stem} version {version.group(1)}{suffix}"
    return f"{suggested_stem}{suffix}"


def _explicit_requested_stem(text: str) -> str:
    span = extract_explicit_filename_span(text)
    return span.display_name if span is not None else ""


def _sanitize_windows_stem(value: str) -> str:
    stem = re.sub(r"\.(?:txt|md|docx|pdf)$", "", value.strip(), flags=re.IGNORECASE)
    return sanitize_windows_stem(stem)


def _conversion_target_format(text: str) -> str:
    resolved = resolve_document_format(text, target_only=True)
    if resolved is not None:
        return resolved.value
    normalized = _normalize(normalize_control_text(text))
    match = re.search(
        r"\b(?:a|en)\s+(?:un\s+)?(?:archivo\s+)?(?:formato\s+)?"
        r"(pdf|docx|word|txt|md|markdown)\b",
        normalized,
    )
    if not match:
        matches = re.findall(r"\b(pdf|docx|word|txt|md|markdown)\b", normalized)
        value = matches[-1] if matches else ""
    else:
        value = match.group(1)
    return {"word": "docx", "md": "markdown"}.get(value, value)


def _extract_rename_destination(text: str) -> str:
    match = re.search(
        r"\b(?:renombr\w*|cambi\w*\s+(?:el\s+)?nombre|modific\w*\s+(?:el\s+)?nombre)"
        r".*?\b(?:a|como)\s+[\"']?(.+?)[\"']?\s*$",
        text,
        re.IGNORECASE,
    )
    if not match:
        return ""
    candidate = re.sub(r"\.(txt|md|docx|pdf)$", lambda item: "." + item.group(1).casefold(), match.group(1).strip(" .\"'"), flags=re.IGNORECASE)
    stem, suffix = (Path(candidate).stem, Path(candidate).suffix) if Path(candidate).suffix else (candidate, "")
    safe = sanitize_windows_stem(stem)
    return f"{safe}{suffix}" if safe else ""


def _extract_path(text: str) -> str:
    match = _WINDOWS_PATH.search(text)
    if match:
        return match.group("path").strip().strip('"')
    for quoted in _QUOTED.findall(text):
        candidate = Path(quoted)
        if candidate.suffix or "\\" in quoted or "/" in quoted:
            return quoted
    return ""


def _extract_filename_entity(text: str) -> str:
    quoted = re.findall(r"[\"']([^\"']+\.(?:txt|md|docx|pdf|py|cpp|zip))[\"']", text, re.IGNORECASE)
    if quoted:
        return Path(quoted[-1]).name
    return _direct_filename_entity(text)


def _direct_filename_entity(text: str) -> str:
    candidates = re.findall(
        r"([A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]"
        r"(?:[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9 _-]{0,178}[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9_-])?"
        r"\.(?:txt|md|docx|pdf|py|cpp|zip))\b",
        text,
        re.IGNORECASE,
    )
    if not candidates:
        return ""
    value = candidates[-1].strip()
    value = re.sub(
        r"^.*\b(?:que\s+se\s+llama|llamad[oa]|con\s+nombre|"
        r"crea|crear|haz|hacer|genera|generar|duplica|duplicar|copia|copiar|"
        r"convierte|convertir|edita|editar|borra|borrar|elimina|eliminar)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^(?:(?:un|el|la)\s+)?(?:(?:archivo|documento)\s+)?"
        r"(?:(?:de\s+)?(?:texto|word)\s+)?(?:que\s+se\s+llama\s+)?",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    return Path(value).name if value else ""


def _referenced_file(normalized: str, state: ConversationToolState) -> str:
    referents = (
        "lo", "este archivo", "ese archivo", "el archivo", "este documento",
        "ese documento", "el documento", "conviertelo",
        "leelo", "editalo", "borralo", "revisalo", "intentalo",
    )
    if any(marker in normalized for marker in referents):
        if "codigo" in normalized and state.active_code_file:
            return state.active_code_file
        if state.active_document_ref is not None:
            return state.active_document_ref.absolute_canonical_path
        return state.active_document
    return ""


def _content_payload(text: str) -> str:
    fenced = re.search(r"```(?:\w+)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    marker = re.search(r"(?:que\s+diga|(?:con|y)\s+(?:este|el)\s+texto)\s*[:,-]?\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
    return marker.group(1).strip() if marker else ""


def _code_payload(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip() + "\n"
    marker = re.search(r"(?:python|codigo)\s*[:,-]\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
    return marker.group(1).strip() + "\n" if marker else ""


def _replacement(text: str) -> tuple[str, str]:
    match = re.search(r"reemplaza\s+[\"'](.+?)[\"']\s+por\s+[\"'](.*?)[\"']", text, re.IGNORECASE | re.DOTALL)
    return (match.group(1), match.group(2)) if match else ("", "")


def _after_marker(text: str, markers: tuple[str, ...]) -> str:
    for marker in markers:
        match = re.search(rf"\b{marker}\b\s*(?:otra\s+linea\s*)?[:,-]?\s*(.*)$", text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip() or "Otra linea"
    return "Otra linea"


def _next_available(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("no_available_document_name")


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _asks_about_pdf(text: str) -> bool:
    return bool(re.search(r"\b(?:que|como)\s+(?:es|funciona|se\s+usa)\s+(?:un\s+)?pdf\b", text))


def _targets_core(text: str) -> bool:
    return "core" in text and any(item in text for item in ("modifica", "cambia", "borra", "edita", "escribe"))


def _video_edit(text: str) -> bool:
    return "video" in text and any(item in text for item in ("corta", "recorta", "edita", "trim"))


def _join_texts(text: str) -> bool:
    return "txt" in text and any(item in text for item in ("une", "unelos", "junta", "combina")) and "pdf" in text


def _compress(text: str) -> bool:
    return "zip" in text and any(item in text for item in ("comprime", "crea", "haz"))


def _delete(text: str) -> bool:
    return any(item in text for item in ("borra", "elimina", "borralo", "eliminalo")) and "historial" not in text


def _run_python(text: str) -> bool:
    return "python" in text and any(item in text for item in ("ejecuta", "corre", "prueba"))


def _patch_code(text: str) -> bool:
    return any(item in text for item in ("corrige", "parchea", "reemplaza")) and any(item in text for item in ("codigo", ".py", ".cpp", "archivo"))


def _append_text(text: str) -> bool:
    return any(item in text for item in ("editalo", "edita este", "agrega", "anade")) and "linea" in text


def _read(text: str) -> bool:
    return any(item in text for item in ("lee ", "leelo", "revisalo", "abre este archivo", "dime de que trata"))


def _error_inquiry(text: str) -> bool:
    return "error" in text and any(item in text for item in ("dime", "que", "explica", "tiene"))
