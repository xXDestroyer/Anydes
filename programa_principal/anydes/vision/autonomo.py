"""
autonomo.py — Coordinador de visión autónoma para Anydes.

Maneja la integración de ConcienciaVisual en la interfaz gráfica del launcher,
asegurando que:
  1. La visión se ejecute en paralelo
  2. Hable de forma espontánea cuando algo la sorprenda
  3. No interfiera con el chat o las interacciones del usuario
  4. Se pueda activar/desactivar desde la interfaz
  5. Reconozca su identidad y evite repetirse
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from anydes.nucleo.config import Config
from anydes.nucleo.orquestador import BehaviorOrchestrator
from anydes.vision.conciencia_visual import ConcienciaVisual
from anydes.vision.memoria_visual import MemoriaVisual
from anydes.vision.identidad import IdentidadAnydes
from anydes.vision.variedad_frases import GeneradorFrasesVariadas
from anydes.vision.observation_store import VisualObservationRecord

logger = logging.getLogger("anydes.vision.autonomo")


class ControladorVisionAutonoma:
    """
    Orquesta la ConcienciaVisual dentro de la interfaz gráfica.
    
    Proporciona:
      - Inicio/parada coordinada
      - Callbacks integrados con la GUI
      - Notificación cuando Anydes está hablando
      - Control de modo gaming
    """

    def __init__(
        self,
        config: Config,
        callback_mostrar: Optional[Callable[[str], None]] = None,
        callback_apend_log: Optional[Callable[[str, str], None]] = None,
        speaking_event: threading.Event | None = None,
        orchestrator: BehaviorOrchestrator | None = None,
    ) -> None:
        """
        Inicializa el coordinador.
        
        Parámetros
        ----------
        config : Config
            Configuración de Anydes (debe tener vision_enabled, etc.)
        callback_mostrar : opcional
            Función que muestra observaciones: def fn(texto: str)
        callback_apend_log : opcional
            Función que agrega al log de la GUI: def fn(speaker, text)
        speaking_event : optional
            Event shared with the voice controller to prevent self-listening during TTS.
        """
        self.config = config
        self.callback_mostrar = callback_mostrar or (lambda t: None)
        self.callback_apend_log = callback_apend_log or (lambda s, t: None)
        self.speaking_event = speaking_event or threading.Event()
        
        self._conciencia: Optional[ConcienciaVisual] = None
        self._activo = False
        
        # Nuevos componentes para humanidad
        self.memoria_visual = MemoriaVisual(max_registros=150)
        self.orchestrator = orchestrator or BehaviorOrchestrator(
            config=config,
            memory=self.memoria_visual,
        )
        self.identidad = IdentidadAnydes()
        self.generador_frases = GeneradorFrasesVariadas(max_historial=20)
        
        # Sensibilidad de reactividad (0.0 = muy callada, 1.0 = muy reactiva)
        self.nivel_reactividad = max(0.0, min(1.0, config.vision_reactivity_level))

    def iniciar(self) -> None:
        """Arranca la visión autónoma si está habilitada."""
        if not self.config.vision_enabled:
            logger.info("Visión autónoma deshabilitada en config")
            return

        if self._activo:
            logger.debug("Visión autónoma ya activa")
            return

        try:
            # Crear instancia de ConcienciaVisual con nuevos componentes
            self._conciencia = ConcienciaVisual(
                memoria_visual=self.memoria_visual,
                identidad=self.identidad,
                generador_frases=self.generador_frases,
                nivel_reactividad=self.nivel_reactividad,
                speaking_event=self.speaking_event,
                orquestador=self.orchestrator,
                config=self.config,
                observation_callback=self._observar_autonomo,
            )
            
            # Arrancar el hilo de visión
            self._conciencia.iniciar()
            self._activo = True
            
            logger.info("ControladorVisionAutonoma iniciado exitosamente")
            self.callback_apend_log(
                "Sistema",
                "👁️ Visión autónoma activada: Anydes observa tu pantalla en tiempo real."
            )
        except Exception as exc:
            logger.error("Error iniciando visión autónoma: %s", exc, exc_info=True)
            self.callback_apend_log(
                "Sistema",
                f"⚠️ Error activando visión: {exc}"
            )

    def detener(self) -> None:
        """Detiene la visión autónoma."""
        if not self._activo or self._conciencia is None:
            return

        try:
            self._conciencia.detener()
            self._activo = False
            logger.info("ControladorVisionAutonoma detenido")
            self.callback_apend_log(
                "Sistema",
                "👁️ Visión autónoma desactivada."
            )
        except Exception as exc:
            logger.error("Error deteniendo visión: %s", exc)

    def esta_activo(self) -> bool:
        """Devuelve si la visión está actualmente activa."""
        return self._activo

    def notificar_hablando(self, hablando: bool) -> None:
        """
        Notifica que Anydes está hablando/respondiendo.
        Evita que la visión la interrumpa mientras ella habla.
        """
        if self._conciencia is not None:
            self._conciencia.notificar_hablando(hablando)

    def activar_modo_gaming(self, activar: bool = True) -> None:
        """Activa modo gaming (reacciones más frecuentes y vivaces)."""
        if self._conciencia is not None:
            self._conciencia.activar_modo_gaming(activar)
    
    def establecer_reactividad(self, nivel: float) -> None:
        """
        Establece el nivel de reactividad (0.0-1.0).
        0.0 = muy callada, 1.0 = muy reactiva
        """
        self.nivel_reactividad = max(0.0, min(1.0, nivel))
        if self._conciencia is not None:
            self._conciencia.nivel_reactividad = self.nivel_reactividad
        logger.info(f"Reactividad de visión: {self.nivel_reactividad:.1%}")
    
    def obtener_prompts_mejorados(self) -> dict[str, str]:
        """Obtiene prompts mejorados con contexto de memoria e identidad."""
        return self.memoria_visual.obtener_prompts_para_modelo(
            profile_name=self.identidad.NOMBRE
        )
    
    def registrar_observacion(
        self,
        descripcion: str,
        categoria: str,
        sentimiento: str = "neutra",
        similitud_anterior: float = 0.0,
    ) -> None:
        """Registra una observación en la memoria visual."""
        self.memoria_visual.agregar(
            descripcion=descripcion,
            categoria=categoria,
            sentimiento=sentimiento,
            similitud_anterior=similitud_anterior,
        )

    def analizar_pantalla_ahora(self, pregunta: str = "") -> str:
        """
        Análisis bajo demanda cuando el usuario pregunta sobre la pantalla.
        """
        if self._conciencia is None:
            return "Visión no está activa."
        return self._conciencia.analizar_ahora(pregunta)

    # ────────────────────────────────────────────────────────────────
    # Callbacks internos (integración con GUI)
    # ────────────────────────────────────────────────────────────────

    def _observar_autonomo(self, observation: VisualObservationRecord) -> None:
        logger.debug("Visual observation available: %s", observation.id)

    def _hablar_autonomo(self, texto: str) -> None:
        _ = texto
        raise RuntimeError("Autonomous vision cannot speak outside ReasoningEngine.")


# ────────────────────────────────────────────────────────────────────
# Factory para crear el coordinador con config ya existente
# ────────────────────────────────────────────────────────────────────

def crear_controlador_vision(
    config: Config,
    callback_mostrar: Optional[Callable[[str], None]] = None,
    callback_apend_log: Optional[Callable[[str, str], None]] = None,
) -> ControladorVisionAutonoma:
    """Factory helper."""
    return ControladorVisionAutonoma(
        config=config,
        callback_mostrar=callback_mostrar,
        callback_apend_log=callback_apend_log,
    )
