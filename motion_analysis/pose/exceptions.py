"""Pose detection errors."""


class PoseError(Exception):
    """Base class for MediaPipe pose failures."""


class PoseInitializationError(PoseError):
    """Raised when the MediaPipe Pose model cannot be created."""


class PoseDetectionError(PoseError):
    """Raised when pose inference fails on a frame."""
