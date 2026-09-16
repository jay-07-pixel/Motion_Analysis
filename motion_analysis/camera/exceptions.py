"""Camera-related exceptions used across capture backends."""


class CameraError(Exception):
    """Base class for camera capture failures."""


class CameraNotFoundError(CameraError):
    """Raised when no compatible camera is connected or discoverable."""


class CameraInitializationError(CameraError):
    """Raised when a camera is found but the stream cannot be started."""


class CameraReadError(CameraError):
    """Raised when an open camera fails to deliver frames."""
