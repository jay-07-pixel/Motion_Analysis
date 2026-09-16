"""Abstract frame source used by the live viewer.

Later backends (uploaded video files, other cameras) should implement this
interface so the display loop does not depend on RealSense APIs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameMetadata:
    """Runtime information about the current frame source.

    Attributes:
        source_name: Human-readable device or file identifier.
        width: Frame width in pixels, reported by the active stream.
        height: Frame height in pixels, reported by the active stream.
        stream_fps: Configured capture rate reported by the source.
    """

    source_name: str
    width: int
    height: int
    stream_fps: float


class FrameSource(ABC):
    """Produces BGR frames for OpenCV display.

    Implementations must be usable as context managers so resources are
    released even when initialization or capture fails mid-run.
    """

    @abstractmethod
    def open(self) -> FrameMetadata:
        """Start the frame source.

        Returns:
            Metadata describing the active stream.

        Raises:
            CameraNotFoundError: If no compatible source is available.
            CameraInitializationError: If the source cannot be started.
        """

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Capture the next color frame.

        Returns:
            A BGR image as a NumPy array with shape (height, width, 3), or
            None when a finite source such as a video file has ended.

        Raises:
            CameraReadError: If a frame cannot be retrieved.
        """

    @abstractmethod
    def get_metadata(self) -> FrameMetadata:
        """Return current stream resolution and FPS.

        Returns:
            Metadata reflecting the active stream, not hard-coded defaults.
        """

    def last_frame_time_seconds(self) -> float | None:
        """Return the timestamp of the last successful ``read()``.

        Live cameras should use the device clock when the SDK provides one.
        Video files should use the file's playback position, or
        ``frame_index / FPS`` when the header reports a positive FPS.
        Sources that have no clock may return None so the app can fall back
        to a monotonic timer. FPS is never hard-coded here.

        Returns:
            Time in seconds, or None if this source has no timestamp yet.
        """
        return None

    @abstractmethod
    def close(self) -> None:
        """Stop capture and release hardware or file handles."""

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
