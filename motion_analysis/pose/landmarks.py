"""2D pose landmarks, extraction, and coordinate conversion.

MediaPipe Pose returns image-normalized coordinates:
    x in [0.0, 1.0]  (fraction of image width)
    y in [0.0, 1.0]  (fraction of image height)

Those values are not pixels. This module converts them using the current
frame size:

    pixel_x = normalized_x * frame_width
    pixel_y = normalized_y * frame_height

MediaPipe may return normalized x or y outside [0, 1] when a joint is
off-screen or inferred beyond the image. The converted pixel values are
kept as-is and later marked in-frame or out-of-frame. They are not clamped.


The motion-analysis pipeline uses PixelKeypoint values only.
"""

from __future__ import annotations

from dataclasses import dataclass

from motion_analysis.pose.exceptions import PoseDetectionError

# Official MediaPipe Pose landmark names, in model index order (33 points).
POSE_LANDMARK_NAMES: tuple[str, ...] = (
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
)

# Main body joints shown on the live overlay for testing.
TEST_LANDMARK_NAMES: tuple[str, ...] = (
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
)


@dataclass(frozen=True)
class NormalizedLandmark:
    """One MediaPipe landmark in normalized image coordinates.

    Attributes:
        name: MediaPipe landmark name, for example LEFT_SHOULDER.
        x: Horizontal position as a fraction of image width. Not pixels.
            May be outside [0, 1] when the joint is beyond the frame.
        y: Vertical position as a fraction of image height. Not pixels.
            May be outside [0, 1] when the joint is beyond the frame.
        visibility: MediaPipe visibility score in [0, 1].
    """

    name: str
    x: float
    y: float
    visibility: float


@dataclass(frozen=True)
class PixelKeypoint:
    """One 2D body landmark in image/pixel coordinates.

    This is the representation used by the motion-analysis system.

    Attributes:
        name: Landmark name matching MediaPipe Pose.
        x: Horizontal pixel coordinate (column) in the current frame.
        y: Vertical pixel coordinate (row) in the current frame.
        visibility: MediaPipe visibility/confidence copied from the landmark.
        in_frame: True when (x, y) lies on a visible pixel of the current
            frame. Out-of-frame points keep their estimated x/y; they are
            not clamped onto the image border.
        visible: True when MediaPipe visibility is at or above the configured
            threshold.
        is_valid: True only when the landmark is both in-frame and visible.
            Invalid landmarks are not drawn and do not update smoothing.
    """

    name: str
    x: float
    y: float
    visibility: float
    in_frame: bool = False
    visible: bool = False
    is_valid: bool = False


@dataclass(frozen=True)
class PoseFrame:
    """Pose output for a single RGB frame.

    Attributes:
        frame_width: Width of the frame used for conversion, in pixels.
        frame_height: Height of the frame used for conversion, in pixels.
        normalized_landmarks: MediaPipe coordinates in the 0-1 image space.
        raw_keypoints: Unfiltered pixel keypoints, including invalid
            estimates that are kept internally.
        smoothed_keypoints: Filtered pixel keypoints. Only those with
            ``is_valid`` should be drawn or displayed.
    """

    frame_width: int
    frame_height: int
    normalized_landmarks: list[NormalizedLandmark]
    raw_keypoints: list[PixelKeypoint]
    smoothed_keypoints: list[PixelKeypoint]

    @property
    def pixel_keypoints(self) -> list[PixelKeypoint]:
        """Smoothed pixel keypoints used for drawing and later analysis.

        Returns:
            The smoothed keypoint list. Raw values remain on ``raw_keypoints``.
        """
        return self.smoothed_keypoints


def extract_landmarks(pose_landmarks: object) -> list[NormalizedLandmark]:
    """Extract named landmarks from a MediaPipe Pose result.

    The returned x and y values stay in MediaPipe's normalized image space.
    Call convert_to_pixel_coordinates() before using them for motion analysis.

    Args:
        pose_landmarks: Either a sequence of MediaPipe landmark objects or an
            object with a `landmark` attribute. Each item must expose `x` and
            `y` in normalized image coordinates.

    Returns:
        One NormalizedLandmark per detected body point, in model index order.

    Raises:
        PoseDetectionError: If the result does not contain a landmark list.
    """
    raw_landmarks = getattr(pose_landmarks, "landmark", pose_landmarks)
    try:
        landmark_list = list(raw_landmarks)
    except TypeError as exc:
        raise PoseDetectionError(
            "MediaPipe Pose result has no landmark list to extract."
        ) from exc
    if not landmark_list:
        raise PoseDetectionError("MediaPipe Pose returned an empty landmark list.")

    extracted: list[NormalizedLandmark] = []
    for index, landmark in enumerate(landmark_list):
        name = (
            POSE_LANDMARK_NAMES[index]
            if index < len(POSE_LANDMARK_NAMES)
            else f"LANDMARK_{index}"
        )
        visibility = getattr(landmark, "visibility", None)
        if visibility is None:
            visibility = getattr(landmark, "presence", None)
        # Missing visibility is treated as 0 so an unknown joint is not drawn.
        extracted.append(
            NormalizedLandmark(
                name=name,
                x=float(landmark.x),
                y=float(landmark.y),
                visibility=float(visibility) if visibility is not None else 0.0,
            )
        )
    return extracted


def normalized_to_pixel(
    x_normalized: float,
    y_normalized: float,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float]:
    """Convert one MediaPipe normalized coordinate to pixel coordinates.

    Args:
        x_normalized: Landmark x as a fraction of image width. May be outside
            [0, 1] if MediaPipe infers a joint beyond the frame.
        y_normalized: Landmark y as a fraction of image height. May be outside
            [0, 1] if MediaPipe infers a joint beyond the frame.
        frame_width: Current frame width in pixels from `frame.shape[1]`.
        frame_height: Current frame height in pixels from `frame.shape[0]`.

    Returns:
        `(pixel_x, pixel_y)` in image coordinates for this frame.

    Raises:
        PoseDetectionError: If the frame size is invalid.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise PoseDetectionError(
            "Cannot convert pose coordinates because the frame size is "
            f"{frame_width}x{frame_height}. Width and height must come from "
            "the current RGB frame."
        )
    return x_normalized * frame_width, y_normalized * frame_height


def pixel_to_draw_xy(x: float, y: float) -> tuple[int, int]:
    """Map a stored sub-pixel landmark to the OpenCV drawing pixel.

    Stored coordinates keep the float result of::

        x_pixel = x_normalized * frame_width
        y_pixel = y_normalized * frame_height

    OpenCV drawing APIs require integers. This rounds to the nearest pixel
    and does **not** scale, flip, crop, or add an offset. Overlay text that
    prints a ``drawn`` coordinate must use this same pair so the printed
    value matches the marker on the displayed image.

    Args:
        x: Stored horizontal pixel coordinate.
        y: Stored vertical pixel coordinate.

    Returns:
        Integer ``(column, row)`` passed to ``cv2.circle`` / ``cv2.line``.
    """
    return int(round(x)), int(round(y))


def convert_to_pixel_coordinates(
    landmarks: list[NormalizedLandmark],
    frame_width: int,
    frame_height: int,
) -> list[PixelKeypoint]:
    """Convert a full set of normalized landmarks to pixel keypoints.

    Frame width and height must be the actual current image size. Do not pass
    hard-coded values such as 1280x720.

    Args:
        landmarks: Normalized MediaPipe landmarks from extract_landmarks().
        frame_width: Current frame width in pixels.
        frame_height: Current frame height in pixels.

    Returns:
        PixelKeypoint list in the same order as `landmarks`.
    """
    keypoints: list[PixelKeypoint] = []
    for landmark in landmarks:
        pixel_x, pixel_y = normalized_to_pixel(
            landmark.x,
            landmark.y,
            frame_width,
            frame_height,
        )
        # Keep values even if they fall outside the image. Validation flags
        # them later instead of clamping onto the frame border.
        keypoints.append(
            PixelKeypoint(
                name=landmark.name,
                x=pixel_x,
                y=pixel_y,
                visibility=landmark.visibility,
            )
        )
    return keypoints
