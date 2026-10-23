from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class ResponseStyleConstraint:
    content_kind: str
    exclusive: bool
    no_explanation: bool

    def prompt_note(self) -> str:
        labels = {
            "formula": "la fórmula o ecuación solicitada",
            "code": "el código solicitado",
            "value": "el valor o resultado solicitado",
        }
        target = labels[self.content_kind]
        compact_math = (
            "\n- Para fórmulas o ecuaciones, entrega una sola expresión matemática compacta; "
            "no desarrolles componentes ni formas equivalentes salvo que se pidan."
            if self.content_kind == "formula"
            else ""
        )
        return (
            "RESTRICCIÓN ESTRUCTURADA DE SALIDA DEL TURNO:\n"
            f"- Devuelve únicamente {target}.\n"
            "- No añadas introducción, explicación, comentario, saludo ni conclusión.\n"
            "- Conserva solamente el contenido pedido por la persona."
            + compact_math
        )


def infer_response_style(user_text: str) -> ResponseStyleConstraint | None:
    normalized = _normalize(user_text)
    exclusive = bool(re.search(r"\b(?:solo|solamente|unicamente|exclusivamente)\b", normalized))
    no_explanation = bool(
        re.search(r"\bsin\s+(?:ninguna\s+)?(?:explicacion|detalles?|comentarios?|introduccion)\b", normalized)
    )
    if not (exclusive or no_explanation):
        return None
    kinds = (
        ("formula", r"\b(?:formula|ecuacion|expresion\s+matematica)\b"),
        ("code", r"\b(?:codigo|script|programa)\b"),
        ("value", r"\b(?:valor|resultado|numero)\b"),
    )
    for kind, pattern in kinds:
        if re.search(pattern, normalized):
            return ResponseStyleConstraint(kind, exclusive, no_explanation)
    return None


def apply_response_style(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], ResponseStyleConstraint | None]:
    if not messages:
        return messages, None
    user_index = next(
        (index for index in range(len(messages) - 1, -1, -1) if messages[index].get("role") == "user"),
        None,
    )
    if user_index is None:
        return messages, None
    current = _current_user_message(str(messages[user_index].get("content") or ""))
    constraint = infer_response_style(current)
    if constraint is None:
        return messages, None
    constrained = [dict(item) for item in messages]
    constrained[user_index]["content"] = (
        str(constrained[user_index].get("content") or "").rstrip()
        + "\n\n"
        + constraint.prompt_note()
    )
    return constrained, constraint


def _current_user_message(content: str) -> str:
    marker = re.search(r"MENSAJE ACTUAL(?:\s+DE\s+[^:]+)?:\s*", content, flags=re.IGNORECASE)
    current = content[marker.end():] if marker else content
    return re.split(r"\n\s*\nCONTINUIDAD INMEDIATA VERIFICADA:", current, maxsplit=1, flags=re.IGNORECASE)[0]


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(text or "").casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))
