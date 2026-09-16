"""Pure 2D motion formulas in pixel coordinates.

All functions operate on smoothed pixel positions ``P = (x, y)``. They do
not know about cameras, videos, or MediaPipe. Callers must pass only valid
coordinates; these helpers do not invent values for missing landmarks.

Formulas:

    Position:               P = (x, y)
    Displacement:           ΔP = P_current - P_initial
    2D displacement mag.:   |ΔP| = sqrt((x_c - x_i)^2 + (y_c - y_i)^2)
    Path segment:           d = sqrt((x_c - x_p)^2 + (y_c - y_p)^2)
    Distance travelled:     sum of path segments between consecutive valid
                            positions
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Point2D:
    """A 2D pixel position.

    Attributes:
        x: Horizontal pixel coordinate (column).
        y: Vertical pixel coordinate (row).
    """

    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        """Return ``(x, y)`` for display or further math."""
        return (self.x, self.y)


def position(x: float, y: float) -> Point2D:
    """Build the current position ``P = (x, y)``.

    Args:
        x: Smoothed pixel x of a valid landmark.
        y: Smoothed pixel y of a valid landmark.

    Returns:
        The position as a Point2D.
    """
    return Point2D(x=float(x), y=float(y))


def displacement_vector(current: Point2D, initial: Point2D) -> Point2D:
    """Straight-line change from the first valid position.

    ``ΔP = P_current - P_initial``

    Args:
        current: Latest valid smoothed position.
        initial: First valid smoothed position for this landmark.

    Returns:
        Pixel displacement ``(Δx, Δy)``.
    """
    return Point2D(x=current.x - initial.x, y=current.y - initial.y)


def displacement_magnitude(current: Point2D, initial: Point2D) -> float:
    """Length of the displacement vector.

    ``|ΔP| = sqrt((x_current - x_initial)^2 + (y_current - y_initial)^2)``

    Args:
        current: Latest valid smoothed position.
        initial: First valid smoothed position for this landmark.

    Returns:
        Non-negative distance in pixels.
    """
    delta = displacement_vector(current, initial)
    return math.hypot(delta.x, delta.y)


def consecutive_distance(previous: Point2D, current: Point2D) -> float:
    """Euclidean distance between two consecutive valid positions.

    Args:
        previous: Smoothed position from the previous valid frame.
        current: Smoothed position from the current valid frame.

    Returns:
        Non-negative segment length in pixels.
    """
    return math.hypot(current.x - previous.x, current.y - previous.y)


def accumulate_distance(
    total_distance: float,
    previous: Point2D,
    current: Point2D,
) -> float:
    """Add one valid path segment to the distance travelled.

    Distance travelled is the running sum of consecutive-frame distances,
    not the straight-line displacement from the start.

    Args:
        total_distance: Sum of previous valid segments, in pixels.
        previous: Smoothed position from the previous valid consecutive frame.
        current: Smoothed position from the current valid frame.

    Returns:
        Updated distance travelled in pixels.
    """
    if total_distance < 0:
        raise ValueError(
            f"Distance travelled cannot be negative, got {total_distance}."
        )
    return total_distance + consecutive_distance(previous, current)
