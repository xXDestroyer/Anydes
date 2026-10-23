"""
variedad_frases.py — Sistema para que Anydes no repita la misma frase.

Mantiene un registro de las últimas frases usadas y permite variar expresiones
para que suene más humana y menos robótica.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from typing import Callable

logger = logging.getLogger("anydes.vision.variedad_frases")


class GeneradorFrasesVariadas:
    """
    Genera frases variadas y evita repeticiones.
    """
    
    # Diferentes formas de iniciar observaciones
    INICIOS_OBSERVACION = [
        "Veo que",
        "Parece que",
        "Mira,",
        "Espera, estás",
        "¿No era que",
        "Ah, claro,",
        "Interesante, estás",
        "Ooh,",
        "Hmm,",
        "Vaya,",
        "Voy a aprovechar y decir que",
        "Por ahí iba,",
    ]
    
    # Reacciones a lo mismo repetido
    REACCIONES_REPETICION = [
        "Sigues en lo mismo.",
        "De nuevo con lo mismo.",
        "Más de lo mismo, eh.",
        "¿Otra vez? jaja",
        "Sigue.",
        "Insistes en esto.",
        "No cambias.",
        "Persistente, ¿eh?",
    ]
    
    # Reacciones a cambios
    REACCIONES_CAMBIO = [
        "¡Cambio de escena!",
        "Ooh, qué giro.",
        "Eso no lo veía venir.",
        "Interesante cambio.",
        "¿Qué hay de nuevo?",
        "¡Vaya!",
        "Ups, cambió.",
        "Momento, ¿dónde estamos ahora?",
    ]
    
    # Reacciones a sorpresas
    REACCIONES_SORPRESA = [
        "¡Espera, qué!",
        "¿Eso qué era?",
        "¡No me lo esperaba!",
        "Wow.",
        "¿En serio?",
        "¡Increíble!",
        "¿Viste eso?",
        "Ajá, claro.",
    ]
    
    # Frases finales/conclusiones
    FINALES = [
        "Interesante.",
        "Bueno, eso está pasando.",
        "Vale.",
        "Así va.",
        "Eso es.",
        "Claro está.",
        "Lo que sea.",
    ]
    
    def __init__(self, max_historial: int = 20):
        self.max_historial = max_historial
        self.ultimas_frases: deque[str] = deque(maxlen=max_historial)
        self.ultimas_categorias: deque[str] = deque(maxlen=10)
    
    def agregar_frase_usada(self, frase: str, categoria: str = "") -> None:
        """Registra una frase que ya se usó."""
        self.ultimas_frases.append(frase.lower().strip())
        if categoria:
            self.ultimas_categorias.append(categoria)
    
    def fue_usada_recientemente(self, frase: str, max_dist: int = 5) -> bool:
        """¿Se usó esta frase hace poco?"""
        frase_clean = frase.lower().strip()
        recientes = list(self.ultimas_frases)[-max_dist:]
        
        for usada in recientes:
            # Comparar similitud simple (no idéntica, sino parecida)
            if self._son_parecidas(frase_clean, usada):
                return True
        
        return False
    
    def _son_parecidas(self, s1: str, s2: str, threshold: float = 0.7) -> bool:
        """¿Dos frases son muy parecidas?"""
        if s1 == s2:
            return True
        
        # Comparación simple: si comparten muchas palabras
        palabras1 = set(s1.split())
        palabras2 = set(s2.split())
        
        if not palabras1 or not palabras2:
            return False
        
        similitud = len(palabras1 & palabras2) / len(palabras1 | palabras2)
        return similitud >= threshold
    
    def obtener_inicio(self, excluir_recientes: bool = True) -> str:
        """Obtiene un inicio variado."""
        opciones = self.INICIOS_OBSERVACION.copy()
        
        if excluir_recientes:
            # Filtrar las que se usaron hace poco
            opciones = [
                o for o in opciones
                if not self.fue_usada_recientemente(o, max_dist=3)
            ]
        
        if not opciones:
            opciones = self.INICIOS_OBSERVACION
        
        elegida = random.choice(opciones)
        self.agregar_frase_usada(elegida, "inicio")
        return elegida
    
    def obtener_reaccion_repeticion(self) -> str:
        """Para cuando Anydes ve lo mismo repetido."""
        opciones = self.REACCIONES_REPETICION.copy()
        opciones = [
            o for o in opciones
            if not self.fue_usada_recientemente(o, max_dist=2)
        ]
        if not opciones:
            opciones = self.REACCIONES_REPETICION
        
        elegida = random.choice(opciones)
        self.agregar_frase_usada(elegida, "repeticion")
        return elegida
    
    def obtener_reaccion_cambio(self) -> str:
        """Para cambios notables."""
        opciones = self.REACCIONES_CAMBIO.copy()
        opciones = [
            o for o in opciones
            if not self.fue_usada_recientemente(o, max_dist=3)
        ]
        if not opciones:
            opciones = self.REACCIONES_CAMBIO
        
        elegida = random.choice(opciones)
        self.agregar_frase_usada(elegida, "cambio")
        return elegida
    
    def obtener_reaccion_sorpresa(self) -> str:
        """Para eventos inesperados."""
        opciones = self.REACCIONES_SORPRESA.copy()
        opciones = [
            o for o in opciones
            if not self.fue_usada_recientemente(o, max_dist=4)
        ]
        if not opciones:
            opciones = self.REACCIONES_SORPRESA
        
        elegida = random.choice(opciones)
        self.agregar_frase_usada(elegida, "sorpresa")
        return elegida
    
    def obtener_final(self) -> str:
        """Para cerrar un comentario."""
        opciones = self.FINALES.copy()
        opciones = [
            o for o in opciones
            if not self.fue_usada_recientemente(o, max_dist=3)
        ]
        if not opciones:
            opciones = self.FINALES
        
        elegida = random.choice(opciones)
        self.agregar_frase_usada(elegida, "final")
        return elegida
    
    def construir_reaccion_inteligente(
        self,
        tipo: str,  # "repeticion", "cambio", "sorpresa", "observacion"
        contexto: str = "",
    ) -> str:
        """
        Construye una reacción coherente basada en el tipo.
        """
        if tipo == "repeticion":
            return self.obtener_reaccion_repeticion()
        elif tipo == "cambio":
            return self.obtener_reaccion_cambio()
        elif tipo == "sorpresa":
            return self.obtener_reaccion_sorpresa()
        elif tipo == "observacion":
            inicio = self.obtener_inicio()
            return f"{inicio} {contexto}" if contexto else inicio
        else:
            return contexto or "Hmm."
    
    def fue_categoria_usada_recientemente(self, categoria: str, max_dist: int = 3) -> bool:
        """¿Se usó recientemente una frase de esta categoría?"""
        recientes = list(self.ultimas_categorias)[-max_dist:]
        return categoria in recientes
