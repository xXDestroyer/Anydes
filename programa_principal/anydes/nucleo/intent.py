from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    name: str
    argument: str = ""


def normalize(text: str) -> str:
    lowered = text.strip().lower()
    decomposed = unicodedata.normalize("NFD", lowered)
    without_accents = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", without_accents).strip()


class IntentMatcher:
    def match(self, text: str) -> Intent | None:
        normalized = normalize(text)
        if not normalized:
            return Intent("empty")

        if normalized in {"ayuda", "help", "que puedes hacer", "que sabes hacer"}:
            return Intent("help")

        if self._has_any(normalized, ["herramientas", "funciones", "comandos disponibles"]):
            return Intent("tools")

        if normalized in {
            "personalidad",
            "prompt",
            "prompt principal",
            "muestra tu personalidad",
            "muestrame tu personalidad",
            "muestra tu prompt",
            "muestrame tu prompt",
            "ver personalidad",
            "ver prompt",
        }:
            return Intent("personality")

        temporary_note = self._extract_after(
            text,
            [
                "por ahora recuerda que",
                "temporalmente recuerda que",
                "para esta sesion recuerda que",
                "guarda temporalmente que",
            ],
        )
        if temporary_note:
            return Intent("tool", f"memory_temporary_write nota={temporary_note}")

        if self._is_memory_profile(normalized):
            return Intent("memory_profile")

        if self._has_any(
            normalized,
            [
                "que hablamos recien",
                "que hemos hablado",
                "ultimas conversaciones",
                "conversaciones recientes",
                "que te dije hace rato",
            ],
        ):
            return Intent("memory_recent")

        if self._has_any(
            normalized,
            [
                "primera palabra",
                "primer palabra",
                "primera cosa que te dije",
                "lo primero que te dije",
                "cuando naciste",
                "cuando nacio",
                "recien nacio",
                "recien naci",
                "primera vez que hablamos",
                "lo primero que dije",
                "mi primera palabra",
            ],
        ):
            return Intent("memory_origin")

        if self._has_any(
            normalized,
            ["borra tu memoria temporal", "limpia tu memoria temporal", "olvida el contexto temporal"],
        ):
            return Intent("tool", "memory_temporary_clear")

        if self._has_any(
            normalized,
            ["memoria temporal", "contexto temporal", "contexto actual", "que recuerdas por ahora"],
        ):
            return Intent("tool", "memory_temporary_read")

        memory_search = self._extract_after(
            text,
            [
                "busca en tus recuerdos",
                "busca en tu memoria",
                "recuerdas algo sobre",
                "que recuerdas sobre",
            ],
        )
        if memory_search:
            return Intent("tool", f"memory_conversation_search {memory_search}")

        if self._has_all(normalized, ["memoria", "permanente"]) or self._has_any(
            normalized,
            ["que recuerdas de mi", "que sabes de mi", "que tienes guardado de mi"],
        ):
            return Intent("tool", "memory_permanent_read")

        permanent_note = self._extract_after(
            text,
            [
                "recuerda que",
                "guarda en tu memoria que",
                "guarda permanentemente que",
                "quiero que recuerdes que",
            ],
        )
        if permanent_note:
            return Intent("tool", f"memory_permanent_write nota={permanent_note}")

        if self._has_any(normalized, ["que hora es", "dime la hora", "fecha actual", "que dia es"]):
            return Intent("tool", "current_time")

        if self._has_any(
            normalized,
            ["informacion del sistema", "datos del sistema", "mi computadora", "info de la pc"],
        ):
            return Intent("tool", "system_info")

        word_text = self._extract_after(text, ["cuenta las palabras de", "cuantas palabras tiene"])
        if word_text:
            return Intent("tool", f"word_count {word_text}")

        json_text = self._extract_after(text, ["formatea este json", "ordena este json", "formatea json"])
        if json_text:
            return Intent("tool", f"format_json {json_text}")

        chrome_search = self._extract_after(
            text,
            [
                "busca en chrome",
                "busca en google",
                "googlea",
                "googlea esto",
                "buscame en google",
            ],
        )
        if chrome_search:
            return Intent("tool", f"chrome_search {chrome_search}")

        youtube_search = self._extract_after(
            text,
            [
                "busca en youtube",
                "pon en youtube",
                "abre en youtube",
            ],
        )
        if youtube_search:
            return Intent("tool", f"youtube_search {youtube_search}")

        if self._has_any(normalized, ["abre youtube", "abrir youtube"]):
            return Intent("tool", "open_youtube")

        if self._wants_spotify(normalized):
            return Intent("tool", "open_spotify")

        if self._has_any(
            normalized,
            [
                "deja de ver mi pantalla",
                "deja de ver la pantalla",
                "para de ver mi pantalla",
                "detener vision de pantalla",
                "deja de compartir pantalla",
            ],
        ):
            return Intent("tool", "screen_watch_stop")

        if self._has_any(
            normalized,
            [
                "comparte pantalla",
                "comparte mi pantalla",
                "empieza a ver mi pantalla",
                "activa vision de pantalla",
                "ve mi pantalla siempre",
            ],
        ):
            return Intent("tool", "screen_watch_start")

        if self._has_any(
            normalized,
            [
                "estado de pantalla",
                "estas viendo mi pantalla",
                "sigues viendo mi pantalla",
            ],
        ):
            return Intent("tool", "screen_status")

        if self._wants_screen_analysis(normalized):
            return Intent("screen_analyze")

        app_name = self._extract_after(text, ["abre la app", "abre aplicacion", "abre ", "abrir "])
        if app_name:
            return Intent("tool", f"open_app {app_name}")

        return None

    def _wants_spotify(self, text: str) -> bool:
        if self._has_any(text, ["abre spotify", "abrir spotify", "pon spotify", "abre el spotify"]):
            return True
        if "spotify" in text:
            return self._has_any(text, ["abre", "abrir", "pon", "abre la app", "abre aplicacion"])
        fuzzy_markers = ("espotif", "sportify", "spotifi", "espotify", "isportify")
        if any(marker in text for marker in fuzzy_markers):
            return self._has_any(text, ["abre", "abrir", "pon"])
        return False

    def _wants_screen_analysis(self, text: str) -> bool:
        if self._has_explicit_screen_analysis(text):
            return True
        return self._wants_screen_advice(text)

    def wants_screen_gaming_advice(self, text: str) -> bool:
        return self._wants_screen_advice(normalize(text))

    def _has_explicit_screen_analysis(self, text: str) -> bool:
        if self._has_any(
            text,
            [
                "mira mi pantalla",
                "mirar mi pantalla",
                "que ves en mi pantalla",
                "que ves en la pantalla",
                "que estas viendo",
                "que estoy haciendo",
                "analiza mi pantalla",
                "describe mi pantalla",
                "captura pantalla",
                "toma captura de pantalla",
            ],
        ):
            return True
        if "pantalla" in text and self._has_any(
            text,
            [
                "te gusta",
                "que opinas",
                "que piensas",
                "como se ve",
                "que ves",
            ],
        ):
            return True
        if self._has_any(text, ["te gusta lo que estoy haciendo", "te gusta lo que hago"]):
            return True
        return False

    def _wants_screen_advice(self, text: str) -> bool:
        advice_phrases = [
            "que deberia comprar",
            "que debo comprar",
            "que comprar",
            "que item comprar",
            "que objeto comprar",
            "que deberia hacer",
            "que debo hacer",
            "que hago aqui",
            "que hago aca",
            "que hago ahora",
            "que deberia elegir",
            "que debo elegir",
            "que elegir",
            "que elijo",
            "que opcion",
            "que build",
            "que hechizo",
            "que skill",
            "que me recomiendas comprar",
            "que me recomiendas",
            "que recomiendas comprar",
            "vale la pena comprar",
            "ayudame aqui",
            "ayudame aca",
            "que item",
            "que objeto",
            "contra quien voy",
            "que camino tomar",
            "por donde ir",
            "deberia pushear",
            "deberia retroceder",
            "deberia pelear",
            "deberia ir",
        ]
        if self._has_any(text, advice_phrases):
            return True
        if self._has_any(text, ["aqui", "aca", "ahora"]) and self._has_any(
            text,
            [
                "comprar",
                "hacer",
                "elegir",
                "recomiendas",
                "consejo",
                "ayuda",
                "item",
                "objeto",
                "build",
            ],
        ):
            return True
        return False

    def _is_memory_profile(self, text: str) -> bool:
        simplified = self._strip_punctuation(text)
        if simplified in {"memoria", "perfil", "cual es tu memoria", "que hay en tu memoria"}:
            return True
        return self._has_any(
            simplified,
            [
                "cual es tu memoria",
                "que hay en tu memoria",
                "muestrame tu memoria",
                "estado de tu memoria",
                "resumen de tu memoria",
                "que tienes en memoria",
                "que guardaste",
            ],
        )

    def _has_any(self, text: str, needles: list[str]) -> bool:
        return any(normalize(needle) in text for needle in needles)

    def _has_all(self, text: str, needles: list[str]) -> bool:
        return all(normalize(needle) in text for needle in needles)

    def _extract_after(self, original_text: str, prefixes: list[str]) -> str:
        normalized_original = normalize(original_text)
        for prefix in prefixes:
            normalized_prefix = normalize(prefix)
            index = normalized_original.find(normalized_prefix)
            if index == -1:
                continue
            start = index + len(normalized_prefix)
            return original_text.strip()[start:].strip(" :.-")
        return ""

    def _strip_punctuation(self, text: str) -> str:
        return re.sub(r"[^\w\s]", "", text).strip()
