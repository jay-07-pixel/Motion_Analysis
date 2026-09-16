"""Uploaded video file as a FrameSource."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from motion_analysis.camera.base import FrameMetadata, FrameSource
from motion_analysis.camera.exceptions import (
    CameraInitializationError,
    CameraReadError,
)

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV is a project requirement
    cv2 = None


class VideoFileSource(FrameSource):
    """Reads BGR frames from a user-selected video file.

    Resolution, FPS, frame count, and duration are read from the file at
    open time. No path, resolution, or FPS value is hard-coded.

    Args:
        file_path: Absolute or relative path chosen through the file picker.
    """

    def __init__(self, file_path: str) -> None:
        self._file_path = Path(file_path)
        self._capture: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[FrameMetadata] = None
        self._frame_index = 0
        self._frame_count = 0
        self._position_seconds = 0.0
        self._duration_seconds = 0.0
        self._ended = False

    def open(self) -> FrameMetadata:
        """Open the video and read its actual resolution and FPS.

        Returns:
            Metadata from the file header, not from hard-coded defaults.

        Raises:
            CameraInitializationError: If OpenCV cannot open the file or the
                file reports an invalid frame size.
        """
        if cv2 is None:
            raise CameraInitializationError(
                "OpenCV is required to read uploaded video files."
            )
        if not self._file_path.is_file():
            raise CameraInitializationError(
                f"Video file not found: {self._file_path}"
            )

        self.close()
        capture = cv2.VideoCapture(str(self._file_path))
        if not capture.isOpened():
            raise CameraInitializationError(
                "Could not open the selected video file. Choose an MP4, AVI, "
                f"MOV, or MKV file that OpenCV can decode. Path: {self._file_path}"
            )

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            capture.release()
            raise CameraInitializationError(
                "The selected video did not report a valid resolution. "
                f"OpenCV returned {width}x{height}."
            )
        if not np.isfinite(fps) or fps < 0:
            fps = 0.0

        self._capture = capture
        self._frame_index = 0
        self._frame_count = max(frame_count, 0)
        self._position_seconds = 0.0
        self._duration_seconds = (
            self._frame_count / fps if fps > 0.0 and self._frame_count > 0 else 0.0
        )
        self._ended = False
        self._metadata = FrameMetadata(
            source_name=self._file_path.name,
            width=width,
            height=height,
            stream_fps=fps,
        )
        return self._metadata

    def read(self) -> np.ndarray | None:
        """Read the next video frame.

        Returns:
            The next BGR frame, or None when the file has ended.

        Raises:
            CameraReadError: If the video is not open.
        """
        if self._capture is None:
            raise CameraReadError("The video file is not open.")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self._ended = True
            return None
        self._ended = False
        self._frame_index = int(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
        position_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
        self._position_seconds = position_ms / 1000.0 if position_ms >= 0 else 0.0
        return frame

    def restart(self) -> None:
        """Seek to the first frame so playback can start again.

        Raises:
            CameraReadError: If the video is not open.
        """
        if self._capture is None:
            raise CameraReadError("The video file is not open.")
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._frame_index = 0
        self._position_seconds = 0.0
        self._ended = False

    def get_metadata(self) -> FrameMetadata:
        """Return the file's reported resolution and FPS.

        Returns:
            Metadata captured when the video was opened.

        Raises:
            CameraReadError: If the video has not been opened.
        """
        if self._metadata is None:
            raise CameraReadError("The video file is not open.")
        return self._metadata

    def last_frame_time_seconds(self) -> Optional[float]:
        """Return this file's playback time for the last decoded frame.

        Prefers OpenCV's reported position in seconds. If that is still 0,
        uses ``(frame_index - 1) / FPS`` from the file header. Unknown FPS
        returns None instead of substituting a hard-coded rate.

        Returns:
            Timestamp in seconds, or None when time cannot be determined.
        """
        if self._capture is None or self._metadata is None:
            return None
        if self._position_seconds > 0.0:
            return self._position_seconds
        fps = self._metadata.stream_fps
        if fps > 0.0 and self._frame_index > 0:
            return (self._frame_index - 1) / fps
        if fps > 0.0:
            return 0.0
        return None

    def playback_status(self) -> dict[str, float | int | bool]:
        """Return dynamic playback position for the overlay.

        Returns:
            Frame index, frame count, time, duration, and whether the file
            has ended. Counts come from the open video, not constants.
        """
        return {
            "frame_index": self._frame_index,
            "frame_count": self._frame_count,
            "position_seconds": self._position_seconds,
            "duration_seconds": self._duration_seconds,
            "ended": self._ended,
        }

    def close(self) -> None:
        """Release the OpenCV capture handle."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._metadata = None
        self._ended = False
