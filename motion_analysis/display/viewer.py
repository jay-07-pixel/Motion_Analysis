"""OpenCV live viewer for RGB frames."""

from __future__ import annotations

from enum import Enum
from collections import deque
from time import perf_counter
from typing import Deque, Optional, Sequence

import cv2
import numpy as np

from motion_analysis.camera.base import FrameMetadata

WINDOW_NAME = "2D Motion Analysis - Pose + Hands"
_QUIT_KEYS = {ord("q"), ord("Q"), 27}  # 27 is the Escape key
_PAUSE_KEYS = {ord(" "), ord("p"), ord("P")}
_RESTART_KEYS = {ord("r"), ord("R")}
_PREV_LANDMARK_KEYS = {ord("["), ord(",")}
_NEXT_LANDMARK_KEYS = {ord("]"), ord(".")}


class ViewerAction(Enum):
    """User action from the OpenCV preview window."""

    CONTINUE = "continue"
    QUIT = "quit"
    TOGGLE_PAUSE = "toggle_pause"
    RESTART = "restart"
    PREV_LANDMARK = "prev_landmark"
    NEXT_LANDMARK = "next_landmark"


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

    def show(
        self,
        frame: np.ndarray,
        metadata: FrameMetadata,
        extra_lines: Optional[Sequence[str]] = None,
        wait_ms: int = 1,
    ) -> ViewerAction:
        """Render one frame and report the user's window action.

        Args:
            frame: BGR image from the active frame source.
            metadata: Actual stream resolution and FPS from the source.
            extra_lines: Optional overlay rows such as video time and controls.
            wait_ms: OpenCV waitKey delay in milliseconds. Video playback
                uses the file FPS; live camera uses 1.

        Returns:
            ViewerAction describing continue, pause, restart, or quit.
        """
        display_fps = self._fps_tracker.tick()
        annotated = _annotate_frame(frame, metadata, display_fps, extra_lines)
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._window_created = True
        cv2.imshow(self.window_name, annotated)
        key = cv2.waitKey(max(1, wait_ms)) & 0xFF
        if key in _QUIT_KEYS:
            return ViewerAction.QUIT
        if key in _PAUSE_KEYS:
            return ViewerAction.TOGGLE_PAUSE
        if key in _RESTART_KEYS:
            return ViewerAction.RESTART
        if key in _PREV_LANDMARK_KEYS:
            return ViewerAction.PREV_LANDMARK
        if key in _NEXT_LANDMARK_KEYS:
            return ViewerAction.NEXT_LANDMARK
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return ViewerAction.QUIT
        return ViewerAction.CONTINUE

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
    extra_lines: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Draw source name, resolution, and FPS onto a copy of the frame.

    Args:
        frame: Original BGR image.
        metadata: Source-reported stream properties.
        display_fps: Measured rendering rate for this viewer.
        extra_lines: Optional extra overlay rows.

    Returns:
        Annotated BGR image ready for imshow.
    """
    annotated = frame.copy()
    fps_label = (
        f"Source FPS: {metadata.stream_fps:.1f}"
        if metadata.stream_fps > 0
        else "Source FPS: unknown"
    )
    lines = [
        metadata.source_name,
        f"Resolution: {metadata.width} x {metadata.height}",
        fps_label,
        f"Display FPS: {display_fps:.1f}",
    ]
    if extra_lines:
        lines.extend(extra_lines)
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
