"""Drawing helpers for live hand and finger landmarks."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from motion_analysis.hands.landmarks import (
    HAND_CONNECTIONS,
    HAND_OVERLAY_NAMES,
    HandFrame,
    HandsFrame,
)
from motion_analysis.pose.landmarks import PixelKeypoint
from motion_analysis.pose.smoothing import SmoothingConfig
from motion_analysis.pose.validation import count_valid, describe_invalid_reason

_LEFT_COLOR = (80, 255, 80)
_RIGHT_COLOR = (0, 165, 255)
_LABEL_COLOR = (255, 255, 255)
_HAND_COLORS = {
    "Left": _LEFT_COLOR,
    "Right": _RIGHT_COLOR,
}


def draw_hand_landmarks(
    frame: np.ndarray,
    hand: HandFrame,
) -> np.ndarray:
    """Draw one hand's valid landmarks, connections, and names.

    A connection is drawn only when both endpoints are valid. Invalid
    coordinates are not clamped onto the image.

    Args:
        frame: BGR image to draw on. Modified in place.
        hand: Pixel-space result for one left or right hand.

    Returns:
        The same frame, with this hand drawn.
    """
    color = _HAND_COLORS.get(hand.handedness, (255, 255, 0))
    points = {
        keypoint.name: keypoint
        for keypoint in hand.smoothed_keypoints
        if keypoint.is_valid
    }
    for start_name, end_name in HAND_CONNECTIONS:
        start = points.get(start_name)
        end = points.get(end_name)
        if start is None or end is None:
            continue
        cv2.line(
            frame,
            _as_drawing_point(start),
            _as_drawing_point(end),
            color,
            2,
            cv2.LINE_AA,
        )
    for keypoint in points.values():
        center = _as_drawing_point(keypoint)
        cv2.circle(frame, center, 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            keypoint.name,
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            color,
            1,
            cv2.LINE_AA,
        )
    wrist = points.get("WRIST")
    if wrist is not None:
        wx, wy = _as_drawing_point(wrist)
        cv2.putText(
            frame,
            hand.handedness,
            (wx - 10, wy - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def draw_hands_overlay(
    frame: np.ndarray,
    hands_frame: HandsFrame,
    smoothing: Optional[SmoothingConfig] = None,
) -> np.ndarray:
    """Draw a summary of valid fingertip coordinates.

    Args:
        frame: BGR image to draw on. Modified in place.
        hands_frame: Detector output for this RGB frame.
        smoothing: Visibility threshold shown in the header.

    Returns:
        The same frame, with a bottom-right hands panel.
    """
    min_visibility = smoothing.min_visibility if smoothing is not None else 0.5
    lines = [
        "Hands: 21 landmarks each (valid pixels only)",
        f"Frame {hands_frame.frame_width}x{hands_frame.frame_height}  "
        f"vis>={min_visibility:.2f}",
    ]
    if not hands_frame.hands:
        lines.append("No hands detected")
        _draw_text_panel(frame, lines, bottom_right=True)
        return frame

    for hand in hands_frame.hands:
        valid_count = count_valid(hand.smoothed_keypoints)
        lines.append(
            f"{hand.handedness}  {valid_count}/21  score={hand.score:.2f}"
        )
        by_name = {keypoint.name: keypoint for keypoint in hand.smoothed_keypoints}
        raw_by_name = {keypoint.name: keypoint for keypoint in hand.raw_keypoints}
        for name in HAND_OVERLAY_NAMES:
            keypoint = by_name.get(name)
            if keypoint is None:
                lines.append(f"  {name}: not detected")
                continue
            if not keypoint.is_valid:
                raw = raw_by_name.get(name, keypoint)
                reason = describe_invalid_reason(raw, min_visibility)
                lines.append(f"  {name}: hidden ({reason})")
                continue
            lines.append(
                f"  {name}  ({keypoint.x:.1f}, {keypoint.y:.1f})  "
                f"vis={keypoint.visibility:.2f}"
            )

    _draw_text_panel(frame, lines, bottom_right=True)
    return frame


def draw_hands_on_frame(
    frame: np.ndarray,
    hands_frame: Optional[HandsFrame],
    error_message: Optional[str] = None,
    smoothing: Optional[SmoothingConfig] = None,
) -> np.ndarray:
    """Add hand landmarks to a (possibly already pose-annotated) frame.

    Args:
        frame: BGR image. Copied before drawing.
        hands_frame: Pixel-space hands result, or None if detection failed
            before producing a frame.
        error_message: Optional per-frame hands error.
        smoothing: Filter settings included in the overlay.

    Returns:
        Annotated copy of the input frame.
    """
    canvas = frame.copy()
    if error_message:
        _draw_text_panel(canvas, ["Hands error", error_message], bottom_right=True)
        return canvas
    if hands_frame is None:
        _draw_text_panel(canvas, ["Hands: no result"], bottom_right=True)
        return canvas

    for hand in hands_frame.hands:
        draw_hand_landmarks(canvas, hand)
    draw_hands_overlay(canvas, hands_frame, smoothing=smoothing)
    return canvas


def _as_drawing_point(keypoint: PixelKeypoint) -> tuple[int, int]:
    """Round a pixel keypoint to an OpenCV drawing coordinate.

    Args:
        keypoint: Landmark already expressed in pixels.

    Returns:
        Integer (x, y) suitable for cv2.line and cv2.circle.
    """
    return int(round(keypoint.x)), int(round(keypoint.y))


def _draw_text_panel(
    frame: np.ndarray,
    lines: list[str],
    bottom_right: bool = False,
) -> None:
    """Draw a semi-transparent text panel onto a frame.

    Args:
        frame: BGR image modified in place.
        lines: Text rows to display.
        bottom_right: If True, place the panel on the bottom-right.
    """
    padding = 8
    line_height = 16
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    thickness = 1
    text_sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    box_width = max(size[0] for size in text_sizes) + padding * 2
    box_height = line_height * len(lines) + padding * 2
    x0 = frame.shape[1] - box_width - 8
    y0 = frame.shape[0] - box_height - 8 if bottom_right else 8
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x0, y0),
        (x0 + box_width, y0 + box_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = y0 + padding + 12
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
