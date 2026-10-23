"""
observadora.py — Anydes observa la pantalla y reacciona con emoción.

Este módulo une ConcienciaVisual + MotorEmocional.
Se arranca desde voice.py con una sola llamada.

Uso desde voice.py o main.py:
    from anydes.vision.observadora import Observadora
    obs = Observadora(tts=mi_funcion_tts, mostrar=mi_funcion_mostrar)
    obs.iniciar()
    ...
    obs.detener()
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

from .conciencia_visual import ConcienciaVisual
from .observation_store import VisualObservationRecord
from ..nucleo.estado_emocional import obtener_motor
from ..nucleo.config import config

logger = logging.getLogger("anydes.vision.observadora")


class Observadora:
    """
    Coordina la visión autónoma con el estado emocional de Anydes.
    Un solo objeto para arrancar/parar todo.
    """

    def __init__(
        self,
        tts: Callable[[str], None] | None = None,
        mostrar: Optional[Callable[[str], None]] = None,
    ):
        self.motor = obtener_motor()
        self.motor.iniciar()

        # Wrapper: al comentar algo visual, actualizar también el estado emocional
        def _hablar_con_emocion(texto: str) -> None:
            self.motor.evento_visual(texto)
            tts(texto)

        def _mostrar_con_emocion(texto: str) -> None:
            print(f"\n💭 Anydes: {texto}")
            if mostrar:
                mostrar(texto)

        def _actualizar_estado(observation: VisualObservationRecord) -> None:
            self.motor.evento_visual(observation.summary)

        self.vision = ConcienciaVisual(
            observation_callback=_actualizar_estado,
            config=config,
        )

    def iniciar(self) -> None:
        self.vision.iniciar()
        logger.info("Observadora iniciada.")

    def detener(self) -> None:
        self.vision.detener()
        self.motor.detener()
        logger.info("Observadora detenida.")

    def notificar_hablando(self, estado: bool) -> None:
        """Avisar que Anydes está hablando (para no interrumpirse)."""
        self.vision.notificar_hablando(estado)

    def activar_gaming(self, activar: bool = True) -> None:
        """Modo gaming: comentarios más frecuentes y vivaces."""
        self.vision.activar_modo_gaming(activar)
        if activar:
            self.motor.cambiar_contexto("gaming")
        else:
            self.motor.cambiar_contexto("general")

    def analizar_ahora(self, pregunta: str = "") -> str:
        """Análisis bajo demanda de la pantalla."""
        return self.vision.analizar_ahora(pregunta)

    def notificar_mensaje(self, mensaje: str, es_usuario: bool = True) -> None:
        """Notificar al motor emocional sobre un intercambio de mensajes."""
        self.motor.evento_conversacion(mensaje, es_usuario=es_usuario)

    def estado_para_prompt(self) -> str:
        """Texto del estado emocional para inyectar en el prompt."""
        return self.motor.como_texto_para_prompt()
