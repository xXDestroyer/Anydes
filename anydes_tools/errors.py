class ToolArchitectureError(RuntimeError):
    """Base error for the isolated tool architecture."""


class PathDeniedError(ToolArchitectureError):
    pass


class PermissionDeniedError(ToolArchitectureError):
    pass


class ToolTimeoutError(ToolArchitectureError):
    pass


class InvalidToolOutputError(ToolArchitectureError):
    pass
