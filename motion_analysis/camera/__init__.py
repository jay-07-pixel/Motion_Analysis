"""Camera backends and shared capture interfaces."""

from motion_analysis.camera.base import FrameMetadata, FrameSource
from motion_analysis.camera.exceptions import (
    CameraError,
    CameraInitializationError,
    CameraNotFoundError,
    CameraReadError,
)
from motion_analysis.camera.realsense import RealSenseColorCamera

__all__ = [
    "CameraError",
    "CameraInitializationError",
    "CameraNotFoundError",
    "CameraReadError",
    "FrameMetadata",
    "FrameSource",
    "RealSenseColorCamera",
]
