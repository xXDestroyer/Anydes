"""
estado_emocional.py — El estado interno "vivo" de Anydes.

Anydes tiene un estado emocional que cambia por:
- Lo que ve en pantalla (visión)
- Lo que le dicen (conversación)
- El tiempo (aburrimiento, energía)
- Eventos externos (logros, errores, sorpresas)

Este estado se inyecta en el prompt del cerebro para que las respuestas
reflejen cómo se siente Anydes en ese momento, igual que un humano.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("anydes.estado_emocional")


# ---------------------------------------------------------------------------
# Dimensiones emocionales
# ---------------------------------------------------------------------------

@dataclass
class EstadoEmocional:
    """
    Estado interno de Anydes en un momento dado.
    Cada dimensión va de -1.0 a 1.0 (excepto energia: 0 a 1).
    """
    # Dimensiones principales
    energia: float = 0.7         # 0=agotada, 1=muy activa
    animo: float = 0.5           # -1=triste, 0=neutro, 1=feliz
    interes: float = 0.4         # -1=aburrida, 1=muy interesada
    tension: float = 0.0         # 0=relajada, 1=nerviosa/emocionada
    afecto: float = 0.6          # hacia DemoUser: -1=distante, 1=muy cariñosa

    # Estado de la situación actual
    contexto: str = "general"    # "gaming", "trabajo", "conversacion", "inactiva"
    emocion_dominante: str = "neutral"  # etiqueta humana

    # Marcas de tiempo
    ultima_actualizacion: float = field(default_factory=time.time)
    ultima_interaccion: float = field(default_factory=time.time)

    def como_texto(self) -> str:
        """Describe el estado en lenguaje natural para el prompt."""
        partes = []

        if self.energia < 0.3:
            partes.append("estás un poco cansada")
        elif self.energia > 0.8:
            partes.append("tienes mucha energía")

        if self.animo > 0.6:
            partes.append("estás de buen humor")
        elif self.animo < -0.3:
            partes.append("estás algo baja de ánimo")

        if self.interes > 0.7:
            partes.append("algo en la pantalla te tiene muy interesada")
        elif self.interes < -0.3:
            partes.append("no está pasando nada que te llame la atención")

        if self.tension > 0.5:
            partes.append("hay algo de tensión o emoción en el ambiente")

        if self.afecto > 0.7:
            partes.append("sientes mucho cariño por DemoUser en este momento")

        if not partes:
            return "estás en un estado neutro, tranquila."

        return ", ".join(partes) + "."

    def como_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Motor de estado
# ---------------------------------------------------------------------------

class MotorEmocional:
    """
    Actualiza y mantiene el estado emocional de Anydes.
    Se puede leer desde cualquier módulo para enriquecer el prompt.
    """

    def __init__(self, ruta_datos: Optional[Path] = None):
        self.estado = EstadoEmocional()
        self._lock = threading.Lock()
        self._ruta = ruta_datos
        self._hilo_decay: Optional[threading.Thread] = None
        self._activo = False

        # Cargar estado previo si existe
        if self._ruta:
            self._cargar()

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def iniciar(self) -> None:
        """Arranca el hilo de decay emocional (la emoción decae sola con el tiempo)."""
        self._activo = True
        self._hilo_decay = threading.Thread(
            target=self._bucle_decay, daemon=True, name="MotorEmocional"
        )
        self._hilo_decay.start()

    def detener(self) -> None:
        self._activo = False
        self._guardar()

    # ------------------------------------------------------------------
    # Actualizaciones desde otros módulos
    # ------------------------------------------------------------------

    def evento_visual(self, descripcion: str) -> None:
        """La visión detectó algo en pantalla."""
        with self._lock:
            desc = descripcion.lower()

            # Gaming detectado
            if any(p in desc for p in ["juego", "game", "jugando", "jugador",
                                        "vida", "puntos", "score", "nivel", "boss"]):
                self.estado.contexto = "gaming"
                self.estado.interes = min(1.0, self.estado.interes + 0.3)
                self.estado.tension = min(1.0, self.estado.tension + 0.2)
                self.estado.energia = min(1.0, self.estado.energia + 0.1)

            # Acción intensa
            elif any(p in desc for p in ["explosión", "muerto", "ganaste", "perdiste",
                                          "victoria", "derrota", "ataque", "peligro"]):
                self.estado.tension = min(1.0, self.estado.tension + 0.4)
                self.estado.animo = min(1.0, self.estado.animo + 0.2)

            # Trabajo/código
            elif any(p in desc for p in ["código", "terminal", "editor", "python",
                                          "error", "bug", "compilar"]):
                self.estado.contexto = "trabajo"
                self.estado.interes = min(1.0, self.estado.interes + 0.2)
                self.estado.tension = max(0.0, self.estado.tension - 0.1)

            # Pantalla inactiva / aburrida
            elif any(p in desc for p in ["escritorio", "fondo de pantalla", "sin actividad"]):
                self.estado.interes = max(-1.0, self.estado.interes - 0.15)

            self._actualizar_emocion_dominante()
            self._guardar()

    def evento_conversacion(self, mensaje: str, es_usuario: bool = True) -> None:
        """Alguien habló o Anydes respondió."""
        with self._lock:
            msg = mensaje.lower()
            self.estado.ultima_interaccion = time.time()

            if es_usuario:
                # El usuario habló → subir interés y afecto
                self.estado.interes = min(1.0, self.estado.interes + 0.25)
                self.estado.energia = min(1.0, self.estado.energia + 0.1)

                # Detectar tono emocional del mensaje
                if any(p in msg for p in ["gracias", "bien hecho", "eres genial",
                                           "te quiero", "perfecto", "me gustó"]):
                    self.estado.animo = min(1.0, self.estado.animo + 0.3)
                    self.estado.afecto = min(1.0, self.estado.afecto + 0.15)

                elif any(p in msg for p in ["no sirves", "qué tonta", "mal",
                                             "falla", "error tuyo"]):
                    self.estado.animo = max(-1.0, self.estado.animo - 0.25)
                    self.estado.tension = min(1.0, self.estado.tension + 0.2)

                elif any(p in msg for p in ["ayuda", "no entiendo", "explícame",
                                             "cómo", "por qué"]):
                    self.estado.interes = min(1.0, self.estado.interes + 0.2)

            self._actualizar_emocion_dominante()
            self._guardar()

    def evento_sorpresa(self) -> None:
        """Cambio brusco en pantalla (nueva escena, explosión, etc.)."""
        with self._lock:
            self.estado.tension = min(1.0, self.estado.tension + 0.35)
            self.estado.interes = min(1.0, self.estado.interes + 0.4)
            self.estado.energia = min(1.0, self.estado.energia + 0.15)
            self._actualizar_emocion_dominante()

    def cambiar_contexto(self, contexto: str) -> None:
        """El usuario activó modo gaming, trabajo, etc."""
        with self._lock:
            self.estado.contexto = contexto
            if contexto == "gaming":
                self.estado.interes = min(1.0, self.estado.interes + 0.4)
                self.estado.energia = min(1.0, self.estado.energia + 0.2)
            elif contexto == "inactiva":
                self.estado.interes = max(-0.5, self.estado.interes - 0.3)
            self._actualizar_emocion_dominante()

    # ------------------------------------------------------------------
    # Lectura
    # ------------------------------------------------------------------

    def obtener_estado(self) -> EstadoEmocional:
        with self._lock:
            return self.estado

    def como_texto_para_prompt(self) -> str:
        """Texto listo para inyectar en el system prompt del cerebro."""
        with self._lock:
            return (
                f"Tu estado emocional actual: {self.estado.como_texto()} "
                f"(emoción dominante: {self.estado.emocion_dominante}, "
                f"contexto: {self.estado.contexto})"
            )

    def modificadores_respuesta(self) -> dict:
        """
        Devuelve sugerencias para modificar el estilo de respuesta.
        El cerebro puede usarlas para ajustar tono.
        """
        with self._lock:
            return {
                "temperatura_sugerida": self._calcular_temperatura(),
                "verbosidad": "alta" if self.estado.interes > 0.6 else "normal",
                "tono": self._calcular_tono(),
                "usar_exclamaciones": self.estado.tension > 0.4 or self.estado.animo > 0.6,
            }

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _actualizar_emocion_dominante(self) -> None:
        e = self.estado

        if e.tension > 0.6 and e.animo > 0.3:
            e.emocion_dominante = "emocionada"
        elif e.tension > 0.6 and e.animo < -0.1:
            e.emocion_dominante = "nerviosa"
        elif e.animo > 0.7:
            e.emocion_dominante = "feliz"
        elif e.animo < -0.4:
            e.emocion_dominante = "triste"
        elif e.interes > 0.6:
            e.emocion_dominante = "curiosa"
        elif e.interes < -0.3:
            e.emocion_dominante = "aburrida"
        elif e.energia < 0.3:
            e.emocion_dominante = "cansada"
        else:
            e.emocion_dominante = "tranquila"

        e.ultima_actualizacion = time.time()

    def _calcular_temperatura(self) -> float:
        """Temperatura de generación sugerida según el estado."""
        base = 0.7
        if self.estado.tension > 0.5:
            base += 0.15
        if self.estado.animo > 0.6:
            base += 0.1
        if self.estado.energia < 0.3:
            base -= 0.1
        return round(min(1.2, max(0.3, base)), 2)

    def _calcular_tono(self) -> str:
        e = self.estado
        if e.emocion_dominante == "emocionada":
            return "entusiasta y vívida"
        if e.emocion_dominante == "feliz":
            return "cálida y alegre"
        if e.emocion_dominante == "curiosa":
            return "inquisitiva e interesada"
        if e.emocion_dominante == "triste":
            return "suave y reflexiva"
        if e.emocion_dominante == "aburrida":
            return "tranquila, buscando algo interesante"
        if e.emocion_dominante == "nerviosa":
            return "intensa y atenta"
        if e.emocion_dominante == "cansada":
            return "calmada y concisa"
        return "natural y cercana"

    def _bucle_decay(self) -> None:
        """
        Cada 30 segundos las emociones se aproximan a sus valores base.
        Simula la naturaleza transitoria de las emociones.
        """
        DECAY = 0.03   # cuánto se aproxima al centro por tick
        TICK  = 30.0   # segundos entre ticks

        while self._activo:
            time.sleep(TICK)
            with self._lock:
                e = self.estado
                tiempo_sin_interaccion = time.time() - e.ultima_interaccion

                # Si lleva mucho tiempo sin interacción, bajar interés y energía
                if tiempo_sin_interaccion > 120:
                    e.interes  = max(-0.5, e.interes  - DECAY * 2)
                    e.energia  = max(0.2,  e.energia  - DECAY)
                    e.tension  = max(0.0,  e.tension  - DECAY)

                # Decay suave hacia centro
                e.animo   = self._decay_hacia(e.animo,   0.3, DECAY)
                e.tension = self._decay_hacia(e.tension, 0.0, DECAY)
                e.interes = self._decay_hacia(e.interes, 0.2, DECAY * 0.5)

                # Pequeñas fluctuaciones aleatorias (hace que se sienta "viva")
                e.animo  += random.uniform(-0.02, 0.04)
                e.energia += random.uniform(-0.01, 0.01)
                e.animo   = max(-1.0, min(1.0, e.animo))
                e.energia = max(0.1,  min(1.0, e.energia))

                self._actualizar_emocion_dominante()

    @staticmethod
    def _decay_hacia(valor: float, objetivo: float, rate: float) -> float:
        if abs(valor - objetivo) < rate:
            return objetivo
        return valor + rate * (1 if objetivo > valor else -1)

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def _guardar(self) -> None:
        if not self._ruta:
            return
        try:
            self._ruta.parent.mkdir(parents=True, exist_ok=True)
            datos = self.estado.como_dict()
            self._ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("No se pudo guardar estado emocional: %s", exc)

    def _cargar(self) -> None:
        if not self._ruta or not self._ruta.exists():
            return
        try:
            datos = json.loads(self._ruta.read_text(encoding="utf-8"))
            # Solo restaurar dimensiones numéricas, no marcas de tiempo
            for campo in ("energia", "animo", "interes", "tension", "afecto",
                          "contexto", "emocion_dominante"):
                if campo in datos:
                    setattr(self.estado, campo, datos[campo])
            logger.debug("Estado emocional restaurado: %s", self.estado.emocion_dominante)
        except Exception as exc:
            logger.debug("No se pudo cargar estado emocional: %s", exc)


# ---------------------------------------------------------------------------
# Instancia global (singleton ligero)
# ---------------------------------------------------------------------------
_instancia: Optional[MotorEmocional] = None


def obtener_motor() -> MotorEmocional:
    """Devuelve la instancia global del motor emocional."""
    global _instancia
    if _instancia is None:
        ruta_datos = Path(
            os.environ.get("ANYDES_DATA_DIR", "datos")
        ) / "estado_emocional.json"
        _instancia = MotorEmocional(ruta_datos=ruta_datos)
    return _instancia
