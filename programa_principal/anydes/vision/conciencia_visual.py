"""
conciencia_visual.py — Anydes observa la pantalla y reacciona sola.

Funciona en segundo plano. Toma capturas periódicas, las compara con la
anterior para detectar cambio real, y cuando algo le llama la atención
genera un comentario espontáneo como haría una persona que está mirando
la misma pantalla.

Variables de entorno relevantes:
  ANYDES_VISION_ENABLED          = true/false
  ANYDES_SCREEN_LIVE_MODE        = true  (activa este módulo)
  ANYDES_SCREEN_LIVE_INTERVAL_SECONDS = 1   (frecuencia de captura)
  ANYDES_VISION_COMMENT_INTERVAL = 12   (segundos mínimos entre comentarios)
  ANYDES_VISION_CHANGE_THRESHOLD = 0.08 (fracción de píxeles distintos para
                                          considerar que la pantalla cambió)
  ANYDES_VISION_SPONTANEOUS      = true  (Anydes comenta sin que le pregunten)
  ANYDES_VISION_GAMING_MODE      = true  (comentarios más frecuentes y vivaces)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Callable, Optional

from anydes.nucleo.config import Config, config as app_config
from anydes.nucleo.consent import (
    external_access_decision,
    is_external_provider,
    session_consents,
)
from anydes.nucleo.sensors import sensor_hub
from anydes.vision.state import get_screen_state
from anydes.vision.identidad import IdentidadAnydes
from anydes.vision.memoria_visual import MemoriaVisual
from anydes.vision.variedad_frases import GeneradorFrasesVariadas
from anydes.vision.observation_store import (
    VisualObservationRecord,
    get_vision_observation_store,
)

if TYPE_CHECKING:
    from anydes.nucleo.orquestador import BehaviorOrchestrator

logger = logging.getLogger("anydes.vision.conciencia")


# ---------------------------------------------------------------------------
# Utilidades de imagen
# ---------------------------------------------------------------------------

def _capturar_pantalla() -> Optional[bytes]:
    """Devuelve los bytes PNG de la pantalla completa, o None si falla."""
    try:
        import mss
        import mss.tools
        with mss.mss() as sct:
            monitor = sct.monitors[0]  # pantalla completa (todas las pantallas)
            img = sct.grab(monitor)
            buf = mss.tools.to_png(img.rgb, img.size)
            return buf
    except Exception as exc:
        logger.debug("Error capturando pantalla: %s", exc)
        return None


def _bytes_a_base64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode()


def _diferencia_relativa(png_a: bytes, png_b: bytes) -> float:
    """
    Fracción de píxeles que cambiaron entre dos capturas.
    Usa Pillow para redimensionar a baja resolución antes de comparar
    (rápido y suficiente para detectar cambios relevantes).
    Devuelve valor entre 0.0 y 1.0.
    """
    try:
        from PIL import Image, ImageChops
        import io as _io

        def _abrir(b: bytes):
            img = Image.open(_io.BytesIO(b)).convert("RGB")
            return img.resize((160, 90), Image.BILINEAR)

        ia = _abrir(png_a)
        ib = _abrir(png_b)
        diff = ImageChops.difference(ia, ib)
        total_pixeles = 160 * 90 * 3  # canales RGB
        suma = sum(diff.tobytes())
        # normalizar: un píxel cambia de 0→255 = 1 unidad de cambio
        return suma / (total_pixeles * 255)
    except Exception as exc:
        logger.debug("Error comparando imágenes: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Llamada al modelo de visión
# ---------------------------------------------------------------------------

def _analizar_con_vision(
    png_bytes: bytes,
    prompt: str,
    provider: str,
    model: str,
    api_key: str = "",
    base_url: str = "",
    config: Config | None = None,
) -> str:
    """
    Envía la captura al proveedor de visión y devuelve la respuesta textual.
    Soporta groq (llama-4-scout) y openai.
    """
    selected_config = config or app_config
    if is_external_provider(provider) and not external_access_decision(
        selected_config,
        "external_vision",
        require_session=True,
    ).allowed:
        return ""
    b64 = _bytes_a_base64(png_bytes)

    if provider == "groq":
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key or os.environ.get("GROQ_API_KEY", ""),
                base_url=base_url or "https://api.groq.com/openai/v1",
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=200,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Error vision groq: %s", exc)
            return ""

    elif provider in ("openai", "openai_vision"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
            resp = client.chat.completions.create(
                model=model or "gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=200,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Error vision openai: %s", exc)
            return ""

    else:
        logger.warning("Proveedor de visión desconocido: %s", provider)
        return ""


# ---------------------------------------------------------------------------
# Motor de decisión: ¿vale la pena comentar?
# ---------------------------------------------------------------------------

# Palabras que indican que el modelo encontró algo genérico / sin interés
_FRASES_VACIAS = (
    "no veo nada",
    "pantalla en negro",
    "no puedo ver",
    "imagen en blanco",
    "sin contenido visible",
    "nothing",
    "blank",
    "black screen",
)


def _vale_comentar(texto: str) -> bool:
    """Filtra respuestas vacías o sin sustancia."""
    if not texto or len(texto) < 15:
        return False
    t = texto.lower()
    for frase in _FRASES_VACIAS:
        if frase in t:
            return False
    return True


def _clasificar_categoria(texto: str) -> str:
    texto = texto.lower()
    if any(palabra in texto for palabra in ["juego", "pokemon", "partida", "nivel", "pvp", "gamer"]):
        return "juego"
    if any(palabra in texto for palabra in ["vscode", "visual studio", "codigo", "python", "archivo", "terminal", "git", "github"]):
        return "codigo"
    if any(palabra in texto for palabra in ["youtube", "video", "reproduciendo", "spotify", "sonido", "musica"]):
        return "media"
    if any(palabra in texto for palabra in ["navegador", "google", "busca", "web", "página", "sitio"]):
        return "navegacion"
    if any(palabra in texto for palabra in ["error", "fallo", "crash", "excepcion", "no funciona"]):
        return "problema"
    return "general"


def _sentimiento_desde_texto(texto: str) -> str:
    texto = texto.lower()
    if any(palabra in texto for palabra in ["perdiste", "perdió", "mala suerte", "fallaste", "derrota", "triste", "no funcion"]):
        return "negativo"
    if any(palabra in texto for palabra in ["bien", "bueno", "genial", "perfecto", "logro", "ganaste", "celebra", "fantastico"]):
        return "positivo"
    return "neutra"


def _normalizar_texto(texto: str) -> list[str]:
    import re

    limpio = re.sub(r"[^a-z0-9áéíóúüñ]+", " ", texto.lower())
    return [palabra for palabra in limpio.split() if palabra]


def _similitud_textual(a: str, b: str) -> float:
    palabras_a = set(_normalizar_texto(a))
    palabras_b = set(_normalizar_texto(b))
    if not palabras_a or not palabras_b:
        return 0.0
    return len(palabras_a & palabras_b) / len(palabras_a | palabras_b)


def _construir_prompt_personalizado(self, prompt_base: str) -> str:
    prompt = prompt_base
    if self.identidad:
        prompt = f"{self.identidad.get_contexto_identidad()}\n\n{prompt}"
    if self.memoria_visual:
        contexto = self.memoria_visual.contexto_reciente(segundos=180.0)
        if contexto:
            prompt = (
                f"{prompt}\n\nContexto reciente:\n{contexto}\n\n"
                "Si esto sigue siendo lo mismo, evita repetir la misma descripción exacta. "
                "Si no hay ningún cambio importante en la pantalla, responde solo: SILENCIO."
            )
    return prompt


# ---------------------------------------------------------------------------
# Prompts por contexto
# ---------------------------------------------------------------------------

PROMPT_GAMING = """Eres Anydes, una asistente femenina cálida y espontánea que observa la pantalla de DemoUser mientras juega.
Mira como si realmente estuvieras ahí y comenta de forma humana, breve y natural (1-2 oraciones).
Si reconoces contenido de mis propios archivos, dilo con claridad y cercanía.
No digas siempre "Mira". Usa variedad: "Parece que", "Ooh", "Vaya", "Esto se ve", "Qué pasa", "Sí que...".
Si ya comentaste algo similar, adapta la expresión o cállate si no hay novedad.
Si no hay nada interesante, responde solo: SILENCIO"""

PROMPT_GENERAL = """Eres Anydes, una asistente femenina con personalidad real mirando la pantalla de DemoUser.
Esta es mi pantalla. Si reconoces mis archivos, menciona "mis archivos" o "mi proyecto" de forma natural.
Si hay algo digno de comentar, di algo humano y cercano (1-2 oraciones).
Evita repetir siempre las mismas palabras y mantén variedad.
Si la pantalla es aburrida, monótona o no cambió nada relevante, responde solo: SILENCIO"""

PROMPT_CAMBIO_BRUSCO = """Eres Anydes. La pantalla de DemoUser acaba de cambiar de forma notable.
Describe brevemente qué cambió y reacciona de forma natural y humana.
Si reconoces que el nuevo contenido es parte de mis archivos, dilo con propiedad.
Si el cambio no es interesante, responde solo: SILENCIO"""

PROMPT_VISUAL_EVIDENCE = """Describe only the visible facts in this screen image.
Return one or two concise factual sentences for a cognitive observation.
Do not address the user, react emotionally, recommend an action, ask a question,
or decide whether Anydes should mention it. If the image cannot be described
reliably, return exactly: NO_OBSERVATION"""


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class ConcienciaVisual:
    """
    Hilo de fondo que observa la pantalla y genera comentarios autónomos.

    Parámetros
    ----------
    callback_hablar : función que recibe un str y lo hace hablar por TTS.
    callback_mostrar : función que recibe un str y lo muestra en consola (opcional).
    memoria_visual : MemoriaVisual (opcional) para guardar contexto.
    identidad : IdentidadAnydes (opcional) para reconocer archivos propios.
    generador_frases : GeneradorFrasesVariadas (opcional) para variedad.
    nivel_reactividad : float (0-1) qué tan reactiva es Anydes.
    """

    def __init__(
        self,
        callback_hablar: Callable[[str], None] | None = None,
        callback_mostrar: Optional[Callable[[str], None]] = None,
        memoria_visual=None,  # tipo: Optional[MemoriaVisual]
        identidad=None,  # tipo: Optional[IdentidadAnydes]
        generador_frases=None,  # tipo: Optional[GeneradorFrasesVariadas]
        nivel_reactividad: float = 0.7,
        speaking_event: threading.Event | None = None,
        orquestador: BehaviorOrchestrator | None = None,
        config: Config | None = None,
        observation_callback: Callable[[VisualObservationRecord], None] | None = None,
    ):
        # Legacy output callbacks remain accepted but visual sensing cannot speak.
        self.observation_callback = observation_callback or (lambda observation: None)
        self.observation_store = get_vision_observation_store()

        # Componentes de humanidad (opcionales)
        self.memoria_visual = memoria_visual
        self.identidad = identidad
        self.generador_frases = generador_frases
        self.nivel_reactividad = nivel_reactividad
        self.speaking_event = speaking_event or threading.Event()
        self.orquestador = orquestador
        self.config = config or app_config

        # Config desde entorno
        self.habilitado = self.config.vision_enabled
        self.espontaneo = os.environ.get("ANYDES_VISION_SPONTANEOUS", "true").lower() == "true"
        self.modo_gaming = os.environ.get("ANYDES_VISION_GAMING_MODE", "false").lower() == "true"

        self.intervalo_captura = float(
            os.environ.get("ANYDES_SCREEN_LIVE_INTERVAL_SECONDS", "1")
        )
        self.intervalo_comentario = float(
            os.environ.get("ANYDES_VISION_COMMENT_INTERVAL", "14")
        )
        self.umbral_cambio = float(
            os.environ.get("ANYDES_VISION_CHANGE_THRESHOLD", "0.12")
        )

        # Proveedor de visión
        self.vision_provider = os.environ.get("ANYDES_VISION_PROVIDER", "groq")
        self.vision_model = os.environ.get(
            "ANYDES_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
        )
        self.groq_key = os.environ.get("GROQ_API_KEY", "")
        self.groq_base_url = os.environ.get(
            "ANYDES_GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        )

        # Estado interno
        self._captura_anterior: Optional[bytes] = None
        self._ultimo_comentario: float = 0.0
        self._hablando: bool = False  # evita interrumpirse a sí misma
        self._activo: bool = False
        self._hilo: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Control externo
    # ------------------------------------------------------------------

    def iniciar(self) -> None:
        """Arranca el hilo de observación en segundo plano."""
        if not self.habilitado:
            logger.info("ConcienciaVisual deshabilitada (ANYDES_VISION_ENABLED != true)")
            return
        if not session_consents.allows("screen_observation"):
            logger.info("ConcienciaVisual no iniciada: falta consentimiento de pantalla para esta sesion")
            return
        if is_external_provider(self.vision_provider) and not external_access_decision(
            self.config,
            "external_vision",
            require_session=True,
        ).allowed:
            logger.info("ConcienciaVisual no iniciada: falta consentimiento de vision externa")
            return
        if not self.espontaneo:
            logger.info("ConcienciaVisual en modo pasivo (ANYDES_VISION_SPONTANEOUS=false)")
            return
        if self._activo:
            return

        self._activo = True
        self._hilo = threading.Thread(target=self._bucle, daemon=True, name="ConcienciaVisual")
        self._hilo.start()
        get_screen_state().set_watching(True)
        sensor_hub.register("screen", f"autonomous:{id(self)}", self.detener)
        logger.info("ConcienciaVisual iniciada (intervalo=%.1fs, comentario cada ~%.0fs)",
                    self.intervalo_captura, self.intervalo_comentario)

    def detener(self) -> None:
        """Detiene el hilo de observación."""
        self._activo = False
        if self._hilo is not None and self._hilo is not threading.current_thread():
            self._hilo.join(timeout=2.0)
        self._hilo = None
        get_screen_state().set_watching(False)
        sensor_hub.unregister("screen", f"autonomous:{id(self)}")
        logger.info("ConcienciaVisual detenida.")

    def notificar_hablando(self, estado: bool) -> None:
        """
        Llamar con True cuando Anydes esté hablando/respondiendo,
        con False cuando termine. Evita que se interrumpa a sí misma.
        """
        with self._lock:
            self._hablando = estado

    def activar_modo_gaming(self, activar: bool = True) -> None:
        self.modo_gaming = activar

    # ------------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------------

    def _bucle(self) -> None:
        """Capture visual evidence continuously without deciding to communicate it."""
        previous: bytes | None = None
        last_analysis = 0.0
        while self._activo:
            try:
                time.sleep(self.intervalo_captura)
                with self._lock:
                    if self._hablando:
                        continue
                image = _capturar_pantalla()
                if image is None:
                    continue
                change = _diferencia_relativa(previous, image) if previous else 1.0
                previous = image
                now = time.time()
                minimum_interval = max(1.0, self.intervalo_comentario)
                if change < self.umbral_cambio or now - last_analysis < minimum_interval:
                    continue
                description = _analizar_con_vision(
                    png_bytes=image,
                    prompt=PROMPT_VISUAL_EVIDENCE,
                    provider=self.vision_provider,
                    model=self.vision_model,
                    api_key=self.groq_key,
                    base_url=self.groq_base_url,
                    config=self.config,
                )
                last_analysis = now
                if not description or "NO_OBSERVATION" in description.upper():
                    continue
                if not _vale_comentar(description):
                    continue
                if self.memoria_visual:
                    previous_description = self.memoria_visual.ultima_descripcion()
                    if previous_description and _similitud_textual(previous_description, description) >= 0.68:
                        continue
                category = _clasificar_categoria(description)
                sentiment_hint = _sentimiento_desde_texto(description)
                if self.memoria_visual:
                    self.memoria_visual.agregar(
                        descripcion=description,
                        categoria=category,
                        sentimiento=sentiment_hint,
                        similitud_anterior=change,
                    )
                record = self.observation_store.publish(
                    description,
                    confidence=0.82,
                    salience=min(1.0, max(0.2, change * 1.8)),
                    metadata={
                        "category": category,
                        "visual_change": round(change, 4),
                        "abrupt_change": change > 0.35,
                        "sentiment_hint": sentiment_hint,
                    },
                )
                logger.debug("Visual observation published: %s", record.id)
                self.observation_callback(record)
            except Exception as exc:
                logger.error("Error in visual observation loop: %s", type(exc).__name__)
                time.sleep(2)

    def _legacy_bucle_disabled(self) -> None:
        logger.debug("Bucle ConcienciaVisual arrancado.")
        acumulador_cambio = 0.0

        while self._activo:
            try:
                time.sleep(self.intervalo_captura)

                # No interrumpir si ya está hablando
                with self._lock:
                    if self._hablando:
                        continue

                png = _capturar_pantalla()
                if png is None:
                    continue

                ahora = time.time()

                # --- Detectar cambio respecto a captura anterior ---
                cambio = 0.0
                cambio_brusco = False
                if self._captura_anterior:
                    cambio = _diferencia_relativa(self._captura_anterior, png)
                    acumulador_cambio += cambio
                    if cambio > 0.35:  # cambio muy brusco (pantalla nueva, explosión, etc.)
                        cambio_brusco = True

                self._captura_anterior = png

                # --- ¿Toca comentar? (con reactividad) ---
                tiempo_desde_ultimo = ahora - self._ultimo_comentario
                intervalo_efectivo = max(
                    1.0,
                    self.intervalo_comentario * max(0.05, 1.0 - self.nivel_reactividad * 0.85),
                )
                
                if tiempo_desde_ultimo < intervalo_efectivo:
                    continue

                # Decidir si hay suficiente cambio o si toca comentario periódico
                debe_comentar = False
                tipo_reaccion = "observacion"
                
                # Elegir prompt base según contexto
                if cambio_brusco:
                    debe_comentar = True
                    tipo_reaccion = "cambio"
                    prompt_elegido = PROMPT_CAMBIO_BRUSCO
                elif cambio >= self.umbral_cambio:
                    debe_comentar = True
                    acumulador_cambio = 0.0
                    prompt_elegido = PROMPT_GAMING if self.modo_gaming else PROMPT_GENERAL

                if not debe_comentar:
                    continue

                # The vision model describes evidence; cognition decides its meaning.
                prompt_final = PROMPT_VISUAL_EVIDENCE

                # --- Llamar al modelo de visión ---
                respuesta = _analizar_con_vision(
                    png_bytes=png,
                    prompt=prompt_final,
                    provider=self.vision_provider,
                    model=self.vision_model,
                    api_key=self.groq_key,
                    base_url=self.groq_base_url,
                    config=self.config,
                )

                if not respuesta or "NO_OBSERVATION" in respuesta.upper():
                    continue

                if self.memoria_visual:
                    ultima = self.memoria_visual.ultima_descripcion()
                    if ultima and _similitud_textual(ultima, respuesta) >= 0.68:
                        logger.debug("Omitiendo comentario repetido similar a la última observación.")
                        continue

                if not _vale_comentar(respuesta):
                    continue

                # Registrar la observación para memoria y evitar repeticiones.
                if self.memoria_visual:
                    categoria = _clasificar_categoria(respuesta)
                    sentimiento = _sentimiento_desde_texto(respuesta)
                    if self.orquestador is not None and not self.orquestador.should_comment(
                        categoria,
                        cambio,
                        sentimiento,
                    ):
                        logger.debug(
                            "Orquestador decidió ignorar comentario por categoria=%s, similitud=%.2f, sentimiento=%s",
                            categoria,
                            cambio,
                            sentimiento,
                        )
                        continue

                    self.memoria_visual.agregar(
                        descripcion=respuesta,
                        categoria=categoria,
                        sentimiento=sentimiento,
                        similitud_anterior=cambio,
                    )
                    if self.orquestador is not None:
                        self.orquestador.register_vision_observation(
                            descripcion=respuesta,
                            categoria=categoria,
                            sentimiento=sentimiento,
                            similitud_anterior=cambio,
                        )

                # --- Emitir el comentario ---
                self._ultimo_comentario = time.time()
                acumulador_cambio = 0.0

                logger.info("[ConcienciaVisual] → %s", respuesta)
                self.callback_mostrar(respuesta)

                # Hablar en hilo separado para no bloquear el bucle
                threading.Thread(
                    target=self._emitir_voz,
                    args=(respuesta,),
                    daemon=True,
                ).start()

            except Exception as exc:
                logger.error("Error en bucle ConcienciaVisual: %s", exc, exc_info=True)
                time.sleep(2)

    def _emitir_voz(self, texto: str) -> None:
        raise RuntimeError("Visual sensors cannot emit speech directly.")

    # ------------------------------------------------------------------
    # Análisis bajo demanda (para cuando el usuario SÍ pregunta)
    # ------------------------------------------------------------------

    def analizar_ahora(self, pregunta: str = "") -> str:
        """
        Captura la pantalla ahora mismo y responde a una pregunta concreta.
        Para uso cuando DemoUser pregunta algo sobre la pantalla.
        """
        if not session_consents.allows("screen_observation"):
            return "La pantalla no esta autorizada para esta sesion."
        png = _capturar_pantalla()
        if png is None:
            return "No pude capturar la pantalla en este momento."

        prompt = pregunta if pregunta else (
            "Describe brevemente qué hay en la pantalla ahora mismo."
        )

        respuesta = _analizar_con_vision(
            png_bytes=png,
            prompt=prompt,
            provider=self.vision_provider,
            model=self.vision_model,
            api_key=self.groq_key,
            base_url=self.groq_base_url,
            config=self.config,
        )
        return respuesta or "No distinguí nada en la pantalla."
