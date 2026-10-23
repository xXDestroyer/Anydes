"""Vision package with lazy compatibility exports."""

__all__ = ["ConcienciaVisual"]


def __getattr__(name: str):
    if name == "ConcienciaVisual":
        from anydes.vision.conciencia_visual import ConcienciaVisual

        return ConcienciaVisual
    raise AttributeError(name)
