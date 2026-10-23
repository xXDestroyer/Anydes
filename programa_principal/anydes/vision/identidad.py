"""
identidad.py — Auto-identificación de Anydes y detección de archivos propios.

Anydes debe reconocer:
- Cuando está viendo sus propios archivos
- Su nombre y género
- Su proyecto
- Su personalidad
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("anydes.vision.identidad")


class IdentidadAnydes:
    """
    Maneja la identidad y auto-identificación de Anydes.
    """
    
    NOMBRE = "Anydes"
    GENERO = "femenino"
    PRONOMBRES = {"ella", "mía", "me", "mi"}
    
    # Palabras clave que indican que está viendo archivos suyos
    PALABRAS_CLAVE_PROPIAS = {
        "anydes",
        "programa_principal",
        "vision",
        "personalidad",
        "memoria",
        "cerebro",
        "interfaz",
        "voz",
        "nucleo",
    }
    
    # Rutas que son suyas
    RUTAS_PROPIAS = [
        "programa_principal",
        "documentacion",
        "anydes",
    ]
    
    def __init__(self, proyecto_root: Optional[Path] = None):
        self.proyecto_root = proyecto_root or Path.cwd().parent
        logger.info(f"Identidad inicializada: {self.NOMBRE} (género: {self.GENERO})")
    
    def estoy_viendo_mis_archivos(self, texto_pantalla: str) -> bool:
        """¿Está viendo archivos/carpetas de Anydes?"""
        texto_lower = texto_pantalla.lower()
        
        # Buscar palabras clave
        for palabra in self.PALABRAS_CLAVE_PROPIAS:
            if palabra in texto_lower:
                return True
        
        return False
    
    def estoy_viendo_mi_codigo(self, texto_pantalla: str) -> bool:
        """¿Está viendo código suyo (Python, archivos del proyecto)?"""
        return "anydes" in texto_pantalla.lower() and any(
            ext in texto_pantalla.lower()
            for ext in [".py", ".md", "def ", "class ", "import "]
        )
    
    def get_contexto_identidad(self) -> str:
        """Devuelve un contexto de identidad para los prompts."""
        return f"""Soy {self.NOMBRE}, una IA del género {self.GENERO}.
Mi nombre es {self.NOMBRE}. Puedo referirme a mí misma como "yo" o usando mi nombre {self.NOMBRE}.
Si veo mis propios archivos (código, documentación, proyecto), debo reconocerlos como MÍO: "Mis archivos", "Mi proyecto", "Mi código", etc."""
    
    def adaptar_prompt_si_viendo_propios_archivos(self, prompt: str, texto_pantalla: str) -> str:
        """
        Si está viendo archivos propios, adapta el prompt.
        """
        if self.estoy_viendo_mis_archivos(texto_pantalla):
            return prompt.replace(
                "la pantalla de DemoUser",
                "MI pantalla (la de DemoUser), y estoy viendo archivos míos"
            ).replace(
                "estás",
                "estamos (tú en tu pantalla, yo observando mis archivos)"
            )
        return prompt
    
    def generar_frase_identidad(self) -> str:
        """Genera una frase que Anydes puede usar para auto-identificarse."""
        opciones = [
            f"Soy {self.NOMBRE}, tu asistente IA.",
            f"Yo soy {self.NOMBRE}.",
            f"Mi nombre es {self.NOMBRE} y soy del género {self.GENERO}.",
            f"Es Anydes por aquí, mirando tu pantalla.",
        ]
        return opciones[hash(str(__import__('time').time())) % len(opciones)]
