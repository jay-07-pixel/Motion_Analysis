"""Drawing helpers for live hand and finger landmarks.

Markers are placed at the stored pixel coordinates of each landmark:

    drawn_x, drawn_y = pixel_to_draw_xy(keypoint.x, keypoint.y)

Those stored floats come from the current displayed frame size:

    x_pixel = x_normalized * frame_width
    y_pixel = y_normalized * frame_height

This module does not resize, crop, flip, or otherwise transform the image
before drawing. If the canvas size does not match the size used for
conversion, landmarks are not scaled onto it (that would invent an offset).
"""

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
from motion_analysis.pose.landmarks import PixelKeypoint, pixel_to_draw_xy
from motion_analysis.pose.smoothing import SmoothingConfig
from motion_analysis.pose.validation import count_valid, describe_invalid_reason

# BGR. RAW is always red so it can be compared with the live fingertip.
_RAW_COLOR = (0, 0, 255)
_SMOOTH_LEFT = (80, 255, 80)
_SMOOTH_RIGHT = (0, 215, 255)
_SELECTED_COLOR = (255, 0, 255)
_LABEL_COLOR = (255, 255, 255)
_SMOOTH_COLORS = {
    "Left": _SMOOTH_LEFT,
    "Right": _SMOOTH_RIGHT,
}


def draw_hand_landmarks(
    frame: np.ndarray,
    hand: HandFrame,
    selected_key: Optional[str] = None,
) -> np.ndarray:
    """Draw all 21 hand landmarks at their stored pixel coordinates.

    RAW detections are red crosses. SMOOTHED points use a different filled
    marker. Names sit next to the RAW point (the detector's actual estimate).
    The selected landmark gets an extra ring at the same pixel as its marker.

    A connection is drawn only when both SMOOTHED endpoints are valid.
    Invalid coordinates are not clamped onto the image.

    Args:
        frame: BGR image to draw on. Modified in place. Must be the same
            resolution used when converting this hand's landmarks.
        hand: Pixel-space result for one left or right hand.
        selected_key: Catalog key such as ``LEFT_HAND:INDEX_TIP``, or None.

    Returns:
        The same frame, with this hand drawn.
    """
    smooth_color = _SMOOTH_COLORS.get(hand.handedness, (255, 255, 0))
    raw_by_name = {point.name: point for point in hand.raw_keypoints}
    smooth_by_name = {point.name: point for point in hand.smoothed_keypoints}

    for start_name, end_name in HAND_CONNECTIONS:
        start = smooth_by_name.get(start_name)
        end = smooth_by_name.get(end_name)
        if start is None or end is None or not start.is_valid or not end.is_valid:
            continue
        cv2.line(
            frame,
            pixel_to_draw_xy(start.x, start.y),
            pixel_to_draw_xy(end.x, end.y),
            smooth_color,
            1,
            cv2.LINE_AA,
        )

    for name in _landmark_draw_order(hand):
        raw = raw_by_name.get(name)
        smooth = smooth_by_name.get(name)
        selected = _is_selected_hand_landmark(hand, name, selected_key)

        if raw is not None and raw.is_valid:
            draw_xy = _draw_raw_marker(frame, raw)
            _draw_landmark_name(frame, name, draw_xy, _RAW_COLOR)
            if selected:
                _draw_selected_ring(frame, raw)

        if smooth is not None and smooth.is_valid:
            _draw_smoothed_marker(frame, smooth, smooth_color)
            if selected:
                _draw_selected_ring(frame, smooth)

    wrist = smooth_by_name.get("WRIST")
    if wrist is not None and wrist.is_valid:
        wx, wy = pixel_to_draw_xy(wrist.x, wrist.y)
        cv2.putText(
            frame,
            hand.handedness,
            (wx - 10, wy - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            smooth_color,
            2,
            cv2.LINE_AA,
        )
    return frame


def draw_hands_overlay(
    frame: np.ndarray,
    hands_frame: HandsFrame,
    smoothing: Optional[SmoothingConfig] = None,
    selected_key: Optional[str] = None,
    selected_label: Optional[str] = None,
    selected_raw: Optional[PixelKeypoint] = None,
    selected_smoothed: Optional[PixelKeypoint] = None,
) -> np.ndarray:
    """Draw a summary of valid fingertip coordinates.

    Args:
        frame: BGR image to draw on. Modified in place.
        hands_frame: Detector output for this RGB frame.
        smoothing: Visibility threshold shown in the header.
        selected_key: Catalog key of the analysed landmark, if any.
        selected_label: Human-readable selected landmark label.
        selected_raw: Raw selected keypoint; printed drawn pixel matches marker.
        selected_smoothed: Smoothed selected keypoint for the same landmark.

    Returns:
        The same frame, with a bottom-right hands panel.
    """
    min_visibility = smoothing.min_visibility if smoothing is not None else 0.5
    lines = [
        "Hands: 21 landmarks each",
        "RAW=red cross  SMOOTH=cyan/green dot",
        f"Frame {hands_frame.frame_width}x{hands_frame.frame_height}  "
        f"vis>={min_visibility:.2f}",
    ]
    selected_lines = _selected_coordinate_lines(
        selected_key,
        selected_label,
        selected_raw,
        selected_smoothed,
    )
    if selected_lines:
        lines.extend(selected_lines)
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
            drawn = pixel_to_draw_xy(keypoint.x, keypoint.y)
            lines.append(
                f"  {name}  stored ({keypoint.x:.1f}, {keypoint.y:.1f})  "
                f"drawn {drawn}"
            )

    _draw_text_panel(frame, lines, bottom_right=True)
    return frame


def draw_hands_on_frame(
    frame: np.ndarray,
    hands_frame: Optional[HandsFrame],
    error_message: Optional[str] = None,
    smoothing: Optional[SmoothingConfig] = None,
    selected_key: Optional[str] = None,
    selected_label: Optional[str] = None,
    selected_raw: Optional[PixelKeypoint] = None,
    selected_smoothed: Optional[PixelKeypoint] = None,
) -> np.ndarray:
    """Add hand landmarks to a (possibly already pose-annotated) frame.

    Drawing uses the stored pixel values with no extra transform. The canvas
    must already be the source RGB frame (plus overlays), not a resized copy.

    Args:
        frame: BGR image. Copied before drawing.
        hands_frame: Pixel-space hands result, or None if detection failed
            before producing a frame.
        error_message: Optional per-frame hands error.
        smoothing: Filter settings included in the overlay.
        selected_key: Catalog key to highlight and print.
        selected_label: Human-readable selected landmark label.
        selected_raw: Raw selected keypoint, if present this frame.
        selected_smoothed: Smoothed selected keypoint, if present.

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

    if (
        canvas.shape[0] != hands_frame.frame_height
        or canvas.shape[1] != hands_frame.frame_width
    ):
        # Do not scale landmarks onto a different resolution; that would
        # offset every marker from the pixels stored for this frame.
        _draw_text_panel(
            canvas,
            [
                "Hands not drawn: frame size mismatch",
                f"canvas {canvas.shape[1]}x{canvas.shape[0]}  "
                f"landmarks {hands_frame.frame_width}x{hands_frame.frame_height}",
            ],
            bottom_right=True,
        )
        return canvas

    for hand in hands_frame.hands:
        draw_hand_landmarks(canvas, hand, selected_key=selected_key)
    draw_hands_overlay(
        canvas,
        hands_frame,
        smoothing=smoothing,
        selected_key=selected_key,
        selected_label=selected_label,
        selected_raw=selected_raw,
        selected_smoothed=selected_smoothed,
    )
    return canvas


def _landmark_draw_order(hand: HandFrame) -> list[str]:
    """Return the 21 landmark names in model index order.

    Args:
        hand: Pixel-space hand result.

    Returns:
        Names to draw. Falls back to whatever keypoints are present.
    """
    names: list[str] = []
    seen: set[str] = set()
    for keypoint in hand.raw_keypoints or hand.smoothed_keypoints:
        if keypoint.name not in seen:
            names.append(keypoint.name)
            seen.add(keypoint.name)
    return names


def _is_selected_hand_landmark(
    hand: HandFrame,
    name: str,
    selected_key: Optional[str],
) -> bool:
    """Return whether this joint is the user-selected analysis target.

    Args:
        hand: Hand that owns the joint.
        name: Landmark name such as ``INDEX_TIP``.
        selected_key: Catalog key such as ``LEFT_HAND:INDEX_TIP``, or None.

    Returns:
        True when the selection is this hand's landmark.
    """
    if not selected_key:
        return False
    side = hand.handedness.strip().lower()
    if side == "left":
        return selected_key == f"LEFT_HAND:{name}"
    if side == "right":
        return selected_key == f"RIGHT_HAND:{name}"
    return False


def _draw_raw_marker(frame: np.ndarray, keypoint: PixelKeypoint) -> tuple[int, int]:
    """Draw a red cross exactly at the stored RAW pixel.

    Args:
        frame: BGR image modified in place.
        keypoint: Valid raw landmark.

    Returns:
        The integer pixel used for the marker (same as printed ``drawn``).
    """
    draw_xy = pixel_to_draw_xy(keypoint.x, keypoint.y)
    cv2.drawMarker(frame, draw_xy, _RAW_COLOR, cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
    cv2.circle(frame, draw_xy, 1, _RAW_COLOR, -1, cv2.LINE_AA)
    return draw_xy


def _draw_smoothed_marker(
    frame: np.ndarray,
    keypoint: PixelKeypoint,
    color: tuple[int, int, int],
) -> tuple[int, int]:
    """Draw the SMOOTHED landmark at its stored pixel.

    Args:
        frame: BGR image modified in place.
        keypoint: Valid smoothed landmark.
        color: BGR color distinct from the red RAW marker.

    Returns:
        The integer pixel used for the marker.
    """
    draw_xy = pixel_to_draw_xy(keypoint.x, keypoint.y)
    cv2.circle(frame, draw_xy, 4, color, 1, cv2.LINE_AA)
    cv2.circle(frame, draw_xy, 1, color, -1, cv2.LINE_AA)
    return draw_xy


def _draw_selected_ring(frame: np.ndarray, keypoint: PixelKeypoint) -> tuple[int, int]:
    """Highlight the selected landmark at the same pixel as its marker.

    Args:
        frame: BGR image modified in place.
        keypoint: Selected raw or smoothed landmark.

    Returns:
        The integer pixel used for the ring.
    """
    draw_xy = pixel_to_draw_xy(keypoint.x, keypoint.y)
    cv2.circle(frame, draw_xy, 10, _SELECTED_COLOR, 2, cv2.LINE_AA)
    return draw_xy


def _draw_landmark_name(
    frame: np.ndarray,
    name: str,
    draw_xy: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Place the landmark name next to the marker, not on a different pixel.

    The text origin is offset a few pixels so it does not cover the marker.
    The marker itself stays at ``draw_xy``.

    Args:
        frame: BGR image modified in place.
        name: Landmark name.
        draw_xy: Integer pixel of the marker.
        color: BGR text color.
    """
    cv2.putText(
        frame,
        name,
        (draw_xy[0] + 6, draw_xy[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        color,
        1,
        cv2.LINE_AA,
    )


def _selected_coordinate_lines(
    selected_key: Optional[str],
    selected_label: Optional[str],
    raw: Optional[PixelKeypoint],
    smoothed: Optional[PixelKeypoint],
) -> list[str]:
    """Build overlay rows that print stored vs drawn selected coordinates.

    Args:
        selected_key: Catalog key, or None.
        selected_label: Display label for that key.
        raw: Raw selected keypoint.
        smoothed: Smoothed selected keypoint.

    Returns:
        Text rows, or an empty list when the selection is not a hand joint.
    """
    if not selected_key or not selected_key.startswith(("LEFT_HAND:", "RIGHT_HAND:")):
        return []
    title = selected_label or selected_key
    lines = [f"Selected {title}"]
    lines.append(_format_selected_point("RAW", raw))
    lines.append(_format_selected_point("SMOOTH", smoothed))
    return lines


def _format_selected_point(label: str, keypoint: Optional[PixelKeypoint]) -> str:
    """Format one selected raw or smoothed coordinate pair.

    Args:
        label: ``RAW`` or ``SMOOTH``.
        keypoint: Landmark to print, or None if missing this frame.

    Returns:
        A single overlay line. Drawn integers come from pixel_to_draw_xy().
    """
    if keypoint is None:
        return f"  {label}: not detected"
    if not keypoint.is_valid:
        return f"  {label}: not drawn (invalid)"
    drawn = pixel_to_draw_xy(keypoint.x, keypoint.y)
    return (
        f"  {label} stored ({keypoint.x:.2f}, {keypoint.y:.2f})  "
        f"drawn {drawn}"
    )


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
