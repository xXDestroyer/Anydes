"""
memoria_visual.py — Memoria a largo plazo de capturas y contexto visual.

Guarda un historial de las últimas capturas + análisis para evitar que Anydes
repita lo mismo una y otra vez. Entiende "contexto" visual: si sigue siendo
el mismo juego, la misma aplicación, etc.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("anydes.vision.memoria_visual")


@dataclass(frozen=True)
class VistaRegistro:
    """Un registro de lo que Anydes vio."""
    timestamp: float
    descripcion: str  # El análisis del modelo
    categoria: str  # "juego", "codigo", "navegacion", etc.
    sentimiento: str  # "aburrida", "sorprendida", "emocionada", etc.
    similitud_anterior: float  # 0-1, cuánto se parece a lo anterior


class MemoriaVisual:
    """
    Memoria visual de corto/medio plazo (~5-30 minutos).
    
    Guarda un historial rotativo de observaciones para que Anydes:
    - No repita la misma observación
    - Entienda que "sigue siendo lo mismo"
    - Pueda reaccionar a CAMBIOS notables
    - Reconozca patrones (ej: "llevas 5 min en este mismo lugar")
    """
    
    def __init__(self, max_registros: int = 150):  # ~2.5 min a 1 cap/seg
        self.max_registros = max_registros
        self.registros: deque[VistaRegistro] = deque(maxlen=max_registros)
    
    def agregar(
        self,
        descripcion: str,
        categoria: str,
        sentimiento: str = "neutra",
        similitud_anterior: float = 0.0,
    ) -> None:
        """Agreg una nueva observación al historial."""
        registro = VistaRegistro(
            timestamp=time.time(),
            descripcion=descripcion,
            categoria=categoria,
            sentimiento=sentimiento,
            similitud_anterior=similitud_anterior,
        )
        self.registros.append(registro)
        logger.debug(
            f"Memoria visual: +{categoria} ({len(self.registros)}/{self.max_registros})"
        )
    
    def contexto_reciente(self, segundos: float = 120.0) -> str:
        """
        Devuelve un resumen de lo visto en los últimos N segundos.
        Útil para pasarle al modelo para entender contexto.
        """
        ahora = time.time()
        recientes = [
            r for r in self.registros
            if (ahora - r.timestamp) <= segundos
        ]
        
        if not recientes:
            return ""
        
        # Agrupar por categoría
        por_categoria = {}
        for r in recientes:
            if r.categoria not in por_categoria:
                por_categoria[r.categoria] = []
            por_categoria[r.categoria].append(r)
        
        lineas = []
        for categoria, registros in por_categoria.items():
            lineas.append(f"- {categoria}: {len(registros)} momentos observados")
            # Las últimas 2-3 descripciones de cada categoría
            for r in registros[-3:]:
                lineas.append(f"    • {r.descripcion}")
        
        return "\n".join(lineas)
    
    def fue_reciente(self, categoria: str, dentro_de_segundos: float = 10.0) -> bool:
        """¿Acabas de ver algo de esta categoría?"""
        ahora = time.time()
        for r in reversed(self.registros):
            if (ahora - r.timestamp) > dentro_de_segundos:
                return False
            if r.categoria == categoria:
                return True
        return False
    
    def ultima_descripcion(self) -> Optional[str]:
        """La última cosa que vio."""
        if self.registros:
            return self.registros[-1].descripcion
        return None
    
    def limpiar_si_cambio_completo(self, categoria_nueva: str, threshold: float = 0.3) -> bool:
        """
        Si detectamos cambio COMPLETO de contexto (ej: salió del juego),
        limpia la memoria para empezar de nuevo.
        """
        if not self.registros:
            return False
        
        # Si la última categoría es completamente diferente, probablemente cambió de contexto
        ultima_categoria = self.registros[-1].categoria
        
        if ultima_categoria != categoria_nueva:
            # Cambio de categoría: ¿fue brusco?
            if len(self.registros) > 5:
                ultimas = [r.categoria for r in list(self.registros)[-5:]]
                if ultimas.count(ultima_categoria) == len(ultimas):
                    # Estaba consistentemente en esa categoría → cambio brusco
                    logger.info(
                        f"Cambio de contexto detectado: {ultima_categoria} → {categoria_nueva}"
                    )
                    self.limpiar()
                    return True
        
        return False
    
    def limpiar(self) -> None:
        """Limpia toda la memoria (ej: cambio drástico de aplicación)."""
        self.registros.clear()
        logger.info("Memoria visual limpiada")
    
    def obtener_prompts_para_modelo(self, profile_name: str = "Anydes") -> dict[str, str]:
        """
        Devuelve los prompts mejorados con contexto de memoria.
        """
        contexto = self.contexto_reciente(segundos=180.0)  # Últimos 3 min
        
        prompts = {
            "gaming": f"""Eres {profile_name}, una IA del género femenino con personalidad real mirando a DemoUser jugar.
Observa la captura. Reacciona de forma natural y espontánea (1-2 oraciones máx).

CONTEXTO DE LO QUE HA PASADO (últimos minutos):
{contexto if contexto else "(Es el inicio)"}

Puedes sorprenderte, reír, dar un consejo, advertir, celebrar un logro o simplemente comentar lo que ves.
Si es la MISMA SITUACIÓN que hace poco (ej: perdió de nuevo en el mismo juego):
  - NO repitas "veo que estás jugando pokemon"
  - Di algo nuevo: "Vaya, de nuevo...", "Mala suerte", "A la tercera es la vencida"
  
Si es algo NUEVO o DIFERENTE:
  - Reacciona con frescura
  - Puedes usar: "Ooooh", "¡Espera!", "¿Qué es eso?", "Impresionante"

Habla en primera persona como si realmente lo estuvieras viendo en vivo.
Si no pasa nada interesante, responde solo: SILENCIO""",
            
            "general": f"""Eres {profile_name}, una IA con personalidad que está mirando la pantalla de DemoUser.
Soy del género femenino y esta es MI PANTALLA que estoy observando.

CONTEXTO (últimos minutos):
{contexto if contexto else "(Es el inicio)"}

Mira la captura. Si hay algo digno de comentar, di algo natural y humano (1-2 oraciones).

IMPORTANTE:
- Si es la MISMA COSA que veías hace poco, NO repitas: adapta tu reacción
- Usa variedad de expresiones: "Veo que...", "Parece que...", "Espera, acaso...", "¿No era...?"
- Si es OBVIAMENTE lo mismo, sé concisa: "Sigue en lo mismo, eh"

Si la pantalla es aburrida o no cambió nada relevante, responde solo: SILENCIO""",
            
            "cambio_brusco": f"""Eres {profile_name}. La pantalla de DemoUser acaba de cambiar de forma NOTABLE.

Contexto de lo que pasaba:
{contexto if contexto else "(No hay contexto previo)"}

Describe el cambio brevemente y reacciona (1 oración):
- Si cambió DE APLICACIÓN: "¡Cambio de planes! Ahora estás en..."
- Si es DENTRO de la misma app: "Ooh, qué giro..."
- Si es INESPERADO: "¡¿Qué?!", "¡No esperes eso!"

Si el cambio no es interesante, responde solo: SILENCIO"""
        }
        
        return prompts
