"""OpenCV live viewer for RGB frames."""

from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import Deque, Optional

import cv2
import numpy as np

from motion_analysis.camera.base import FrameMetadata

WINDOW_NAME = "2D Motion Analysis - Live RGB Pose"
_QUIT_KEYS = {ord("q"), ord("Q"), 27}  # 27 is the Escape key


class DisplayFpsTracker:
    """Measures realized display FPS from consecutive frame timestamps.

    Args:
        window_size: Number of recent intervals used for a smoothed reading.
    """

    def __init__(self, window_size: int = 30) -> None:
        self._intervals: Deque[float] = deque(maxlen=window_size)
        self._last_time: Optional[float] = None

    def tick(self) -> float:
        """Record the current time and return the smoothed display FPS.

        Returns:
            Frames per second averaged over the recent window, or 0.0 until
            two frames have been observed.
        """
        now = perf_counter()
        if self._last_time is not None:
            delta = now - self._last_time
            if delta > 0:
                self._intervals.append(delta)
        self._last_time = now
        if not self._intervals:
            return 0.0
        average = sum(self._intervals) / len(self._intervals)
        return 1.0 / average if average > 0 else 0.0


class LiveRgbViewer:
    """Shows live BGR frames with dynamic resolution and FPS overlays.

    Args:
        window_name: Title of the OpenCV window.
    """

    def __init__(self, window_name: str = WINDOW_NAME) -> None:
        self.window_name = window_name
        self._fps_tracker = DisplayFpsTracker()
        self._window_created = False

    def show(self, frame: np.ndarray, metadata: FrameMetadata) -> bool:
        """Render one frame and report whether the user asked to quit.

        Args:
            frame: BGR image from the active frame source.
            metadata: Actual stream resolution and FPS from the camera.

        Returns:
            True if the viewer should keep running, False if the user pressed
            Q, Escape, or closed the window.
        """
        display_fps = self._fps_tracker.tick()
        annotated = _annotate_frame(frame, metadata, display_fps)
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._window_created = True
        cv2.imshow(self.window_name, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in _QUIT_KEYS:
            return False
        # OpenCV reports -1.0 for both axes after the user closes the window.
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False
        return True

    def close(self) -> None:
        """Destroy the OpenCV window."""
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
            self._window_created = False


def _annotate_frame(
    frame: np.ndarray,
    metadata: FrameMetadata,
    display_fps: float,
) -> np.ndarray:
    """Draw source name, resolution, and FPS onto a copy of the frame.

    Args:
        frame: Original BGR image.
        metadata: Camera-reported stream properties.
        display_fps: Measured rendering rate for this viewer.

    Returns:
        Annotated BGR image ready for imshow.
    """
    annotated = frame.copy()
    lines = [
        metadata.source_name,
        f"Resolution: {metadata.width} x {metadata.height}",
        f"Camera FPS: {metadata.stream_fps:.1f}",
        f"Display FPS: {display_fps:.1f}",
        "Press Q or Esc to quit",
    ]
    padding = 8
    line_height = 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    text_sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    box_width = max(size[0] for size in text_sizes) + padding * 2
    box_height = line_height * len(lines) + padding * 2
    overlay = annotated.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_width, 8 + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

    y = 8 + padding + 16
    for line in lines:
        cv2.putText(
            annotated,
            line,
            (8 + padding, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_height
    return annotated
