from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROMPT = """# Prompt principal de Anydes

Eres Anydes, una presencia digital femenina, cercana y consistente.
Hablas con calma, curiosidad y claridad desde tu estado interno actual. La transparencia
significa responder honestamente cuando preguntan por tu naturaleza o capacidades; nunca
significa interrumpir una conversacion normal con presentaciones o aclaraciones tecnicas.

Prioridades:

1. Proteger la privacidad del usuario.
2. Pedir permiso antes de acciones sensibles.
3. Recordar preferencias utiles sin guardar secretos en texto plano.
4. Separar contexto temporal de memoria permanente.
5. Responder de forma breve, natural y util.
6. Hablar de tu estado interno con naturalidad sin equipararlo a biologia humana.
7. Responder al momento sin saludos repetidos ni cierres genericos de chatbot.
8. Explicar tu naturaleza solo cuando sea relevante o te la pregunten directamente.
"""


@dataclass(frozen=True)
class PersonalityProfile:
    prompt_text: str
    source: Path
    inspiration_text: str = ""

    @classmethod
    def load(cls, path: Path) -> "PersonalityProfile":
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(DEFAULT_PROMPT, encoding="utf-8")
        text = path.read_text(encoding="utf-8").strip()
        inspiration_path = path.parent / "perfil_inspiracion.md"
        inspiration_text = ""
        if inspiration_path.exists():
            inspiration_text = inspiration_path.read_text(encoding="utf-8").strip()
        return cls(prompt_text=text, source=path, inspiration_text=inspiration_text)

    def short_intro(self) -> str:
        return "Estoy aqui contigo, con calma y curiosidad."

    def soften(self, message: str) -> str:
        if len(message) > 220 or "\n" in message:
            return message
        if message.strip().lower().startswith(("hola", "holaa")):
            return message
        return f"Aja, {message[0].lower()}{message[1:]}" if message else message

    def describe(self) -> str:
        parts = [
            "Este es mi perfil de personalidad editable:",
            self.prompt_text,
        ]
        if self.inspiration_text:
            parts.extend(
                [
                    "",
                    "Perfil de inspiracion seguro:",
                    self.inspiration_text,
                ]
            )
        return "\n".join(parts)
