"""Hand detection errors."""


class HandsError(Exception):
    """Base class for MediaPipe Hands failures."""


class HandsInitializationError(HandsError):
    """Raised when the MediaPipe Hand Landmarker cannot be created."""


class HandsDetectionError(HandsError):
    """Raised when hand inference fails on a frame."""
