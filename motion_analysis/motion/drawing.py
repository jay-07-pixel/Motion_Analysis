"""Overlay for the selected landmark's 2D motion measurements.

Highlight markers use ``pixel_to_draw_xy`` so the printed ``drawn`` pixel is
the same integer passed to OpenCV. No extra scale, crop, or flip is applied.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from motion_analysis.motion.geometry import Point2D
from motion_analysis.motion.tracker import MotionMeasurement
from motion_analysis.pose.landmarks import PixelKeypoint, pixel_to_draw_xy

_HIGHLIGHT_COLOR = (255, 0, 255)
_RAW_COLOR = (0, 0, 255)
_LABEL_COLOR = (255, 255, 255)


def draw_motion_analysis(
    frame: np.ndarray,
    measurement: MotionMeasurement,
    selected_keypoint: Optional[PixelKeypoint] = None,
    selected_raw: Optional[PixelKeypoint] = None,
) -> np.ndarray:
    """Copy a pose/hands frame and add the selected-landmark readout.

    Args:
        frame: BGR image already annotated with skeleton and hands.
        measurement: Snapshot for the user-selected landmark.
        selected_keypoint: Current smoothed keypoint, highlighted when valid.
        selected_raw: Current raw keypoint, highlighted in red when valid.

    Returns:
        Annotated copy. The input frame is not modified.
    """
    canvas = frame.copy()
    if selected_raw is not None and selected_raw.is_valid:
        _highlight_keypoint(canvas, selected_raw, _RAW_COLOR)
    if selected_keypoint is not None and selected_keypoint.is_valid:
        _highlight_keypoint(canvas, selected_keypoint, _HIGHLIGHT_COLOR)
    _draw_text_panel(
        canvas,
        _measurement_lines(measurement, selected_raw, selected_keypoint),
    )
    return canvas


def _measurement_lines(
    measurement: MotionMeasurement,
    selected_raw: Optional[PixelKeypoint],
    selected_smoothed: Optional[PixelKeypoint],
) -> list[str]:
    """Build overlay rows for position, displacement, and path length.

    Args:
        measurement: Snapshot produced by MotionTracker.
        selected_raw: Raw selected keypoint for coordinate printout.
        selected_smoothed: Smoothed selected keypoint for coordinate printout.

    Returns:
        Text lines for the bottom-left panel.
    """
    fps_text = (
        f"{measurement.source_fps:.2f}"
        if measurement.source_fps > 0
        else "unknown"
    )
    lines = [
        "2D motion (smoothed pixels)",
        f"Landmark: {measurement.landmark.label}",
        f"t = {measurement.timestamp_seconds:.3f}s   source FPS = {fps_text}",
        "[ / ] change landmark",
        _format_selected_pixels("RAW", selected_raw),
        _format_selected_pixels("SMOOTH", selected_smoothed),
    ]
    if not measurement.is_valid:
        lines.append(f"Position: not calculated ({measurement.invalid_reason})")
        lines.append("Displacement: not calculated")
        lines.append(
            f"Distance travelled: {measurement.distance_travelled:.1f} px "
            f"({measurement.sample_count} valid samples)"
        )
        return lines

    point = measurement.position
    assert point is not None
    lines.append(f"Position P = {_format_point(point)} px")
    if measurement.initial_position is not None:
        lines.append(
            f"Initial P = {_format_point(measurement.initial_position)} px"
        )
    if measurement.displacement is not None and measurement.displacement_magnitude is not None:
        lines.append(
            f"Displacement ΔP = {_format_point(measurement.displacement)} px"
        )
        lines.append(f"|ΔP| = {measurement.displacement_magnitude:.1f} px")
    lines.append(
        f"Distance travelled = {measurement.distance_travelled:.1f} px "
        f"({measurement.sample_count} valid samples)"
    )
    return lines


def _format_selected_pixels(label: str, keypoint: Optional[PixelKeypoint]) -> str:
    """Print stored floats and the exact integer pixel used for the marker.

    Args:
        label: ``RAW`` or ``SMOOTH``.
        keypoint: Selected landmark, or None if it is missing this frame.

    Returns:
        One overlay line. ``drawn`` is ``pixel_to_draw_xy(x, y)``.
    """
    if keypoint is None:
        return f"{label}: not detected"
    if not keypoint.is_valid:
        return f"{label}: not drawn (invalid)"
    drawn = pixel_to_draw_xy(keypoint.x, keypoint.y)
    return (
        f"{label} stored ({keypoint.x:.2f}, {keypoint.y:.2f})  "
        f"drawn {drawn}"
    )


def _format_point(point: Point2D) -> str:
    """Format a pixel point for the overlay.

    Args:
        point: Position or displacement vector.

    Returns:
        ``(x, y)`` with one decimal place.
    """
    return f"({point.x:.1f}, {point.y:.1f})"


def _highlight_keypoint(
    frame: np.ndarray,
    keypoint: PixelKeypoint,
    color: tuple[int, int, int],
) -> None:
    """Mark the analysed joint at the same pixel as its hand/body marker.

    Args:
        frame: BGR image modified in place.
        keypoint: Valid landmark whose stored x/y are used with no extra offset.
        color: BGR highlight color.
    """
    center = pixel_to_draw_xy(keypoint.x, keypoint.y)
    cv2.circle(frame, center, 10, color, 2, cv2.LINE_AA)
    cv2.circle(frame, center, 1, color, -1, cv2.LINE_AA)


def _draw_text_panel(frame: np.ndarray, lines: list[str]) -> None:
    """Draw the motion panel at the bottom-left of the frame.

    Args:
        frame: BGR image modified in place.
        lines: Text rows to display.
    """
    padding = 8
    line_height = 18
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    text_sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    box_width = max(size[0] for size in text_sizes) + padding * 2
    box_height = line_height * len(lines) + padding * 2
    x0 = 8
    y0 = frame.shape[0] - box_height - 8
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x0, y0),
        (x0 + box_width, y0 + box_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = y0 + padding + 13
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x0 + padding, y),
            font,
            scale,
            _LABEL_COLOR,
            thickness,
            cv2.LINE_AA,
        )
        y += line_height
