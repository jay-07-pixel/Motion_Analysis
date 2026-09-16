"""Shared 2D frame processing used by live camera and uploaded video.

Both input types must call this module so pose, hands, validation,
smoothing, and 2D motion measurements stay identical after an RGB frame
is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from motion_analysis.hands import (
    HandsDetectionError,
    HandsEstimator,
    draw_hands_on_frame,
)
from motion_analysis.hands.landmarks import HandsFrame
from motion_analysis.motion.landmarks import (
    KeypointObservation,
    LandmarkTarget,
    collect_observations,
    cycle_landmark,
    default_landmark,
)
from motion_analysis.motion.tracker import MotionMeasurement, MotionTracker
from motion_analysis.pose import (
    PoseDetectionError,
    PoseEstimator,
    SmoothingConfig,
    draw_pose_on_frame,
)
from motion_analysis.pose.landmarks import PixelKeypoint, PoseFrame


@dataclass
class ProcessedFrame:
    """Result of the common 2D pipeline for one RGB frame.

    Attributes:
        annotated: Frame with valid body and hand overlays drawn.
        pose_frame: Body pose result, or None if nobody was detected.
        hands_frame: Hands result, or an empty HandsFrame if none were found.
        pose_error: Per-frame pose error message, if inference failed.
        hands_error: Per-frame hands error message, if inference failed.
        measurement: 2D motion snapshot for the selected landmark.
        timestamp_seconds: Source timestamp used for this frame.
        source_fps: FPS reported by the live camera or video file.
    """

    annotated: np.ndarray
    pose_frame: Optional[PoseFrame]
    hands_frame: Optional[HandsFrame]
    pose_error: Optional[str]
    hands_error: Optional[str]
    measurement: Optional[MotionMeasurement] = None
    timestamp_seconds: float = 0.0
    source_fps: float = 0.0


def process_rgb_frame(
    frame: np.ndarray,
    pose_estimator: PoseEstimator,
    hands_estimator: HandsEstimator,
    smoothing: SmoothingConfig,
    selected_landmark: Optional[LandmarkTarget] = None,
) -> ProcessedFrame:
    """Run the shared 2D pipeline on one RGB frame.

    Live RealSense frames and uploaded video frames both enter here:

        RGB → Pose + Hands → validation → smoothing → 2D pixel keypoints

    The displayed canvas is a copy of ``frame`` at the same width/height.
    Landmark pixels are converted from that size, then drawn with
    ``pixel_to_draw_xy`` so markers sit on the stored coordinates.

    Args:
        frame: BGR image from any FrameSource. Not resized before detection
            or display.
        pose_estimator: Initialized MediaPipe Pose estimator.
        hands_estimator: Initialized MediaPipe Hands estimator.
        smoothing: Shared validity and EMA settings.
        selected_landmark: Optional catalog target to highlight on hands.

    Returns:
        Annotated image plus internal raw/smoothed pose and hand results.
    """
    pose_frame = None
    pose_error = None
    hands_frame = None
    hands_error = None
    try:
        pose_frame = pose_estimator.estimate(frame)
    except PoseDetectionError as exc:
        pose_error = str(exc)
    try:
        hands_frame = hands_estimator.estimate(frame)
    except HandsDetectionError as exc:
        hands_error = str(exc)

    observations = collect_observations(pose_frame, hands_frame)
    selected_obs = (
        observations.get(selected_landmark.key) if selected_landmark is not None else None
    )

    annotated = draw_pose_on_frame(
        frame,
        pose_frame,
        error_message=pose_error,
        smoothing=smoothing,
    )
    annotated = draw_hands_on_frame(
        annotated,
        hands_frame,
        error_message=hands_error,
        smoothing=smoothing,
        selected_key=selected_landmark.key if selected_landmark is not None else None,
        selected_label=selected_landmark.label if selected_landmark is not None else None,
        selected_raw=selected_obs.raw if selected_obs is not None else None,
        selected_smoothed=selected_obs.smoothed if selected_obs is not None else None,
    )
    return ProcessedFrame(
        annotated=annotated,
        pose_frame=pose_frame,
        hands_frame=hands_frame,
        pose_error=pose_error,
        hands_error=hands_error,
    )


class FrameProcessor:
    """Owns pose, hands, and 2D motion tracking for the shared RGB pipeline.

    Live camera and uploaded video both call ``process()`` so detection
    and measurement code is not duplicated per input type.
    """

    def __init__(
        self,
        smoothing: SmoothingConfig | None = None,
        selected_landmark: LandmarkTarget | None = None,
    ) -> None:
        self.smoothing = smoothing or SmoothingConfig()
        self.selected_landmark = selected_landmark or default_landmark()
        self.pose_estimator = PoseEstimator(smoothing=self.smoothing)
        self.hands_estimator = HandsEstimator(smoothing=self.smoothing)
        self.motion_tracker = MotionTracker(min_visibility=self.smoothing.min_visibility)
        self._frame_index = 0

    def process(
        self,
        frame: np.ndarray,
        timestamp_seconds: float,
        source_fps: float = 0.0,
    ) -> ProcessedFrame:
        """Process one RGB frame through the shared 2D pipeline.

        Path for both inputs:

            RGB → Pose + Hands → validation → smoothing → 2D keypoints
            → position / displacement / distance travelled

        Args:
            frame: BGR image from the live camera or an uploaded video.
            timestamp_seconds: Actual frame time from the source clock or
                file position. Not a hard-coded FPS.
            source_fps: FPS reported by the active source, or 0 if unknown.

        Returns:
            Annotated frame, keypoints, and the selected landmark measurement.
        """
        result = process_rgb_frame(
            frame,
            self.pose_estimator,
            self.hands_estimator,
            self.smoothing,
            selected_landmark=self.selected_landmark,
        )
        self._frame_index += 1
        self.motion_tracker.update(
            pose_frame=result.pose_frame,
            hands_frame=result.hands_frame,
            timestamp_seconds=timestamp_seconds,
            source_fps=source_fps,
            frame_index=self._frame_index,
            smoothing=self.smoothing,
        )
        result.timestamp_seconds = timestamp_seconds
        result.source_fps = source_fps
        result.measurement = self.motion_tracker.snapshot(self.selected_landmark)
        return result

    def current_measurement(self) -> MotionMeasurement:
        """Return the selected landmark's latest motion snapshot.

        Returns:
            Values from the last ``process()`` call for the current target.
        """
        return self.motion_tracker.snapshot(self.selected_landmark)

    def selected_observation(
        self, result: ProcessedFrame
    ) -> Optional[KeypointObservation]:
        """Return raw and smoothed pixels for the selected landmark.

        Args:
            result: Latest shared-pipeline output.

        Returns:
            The observation pair, or None if that landmark is absent. Missing
            detections are not an error; the overlay prints "not detected".
        """
        observations = collect_observations(result.pose_frame, result.hands_frame)
        return observations.get(self.selected_landmark.key)

    def selected_keypoint(self, result: ProcessedFrame) -> Optional[PixelKeypoint]:
        """Find the current smoothed keypoint for the selected landmark.

        Args:
            result: Latest shared-pipeline output.

        Returns:
            The smoothed PixelKeypoint, or None if it is absent this frame.
        """
        observation = self.selected_observation(result)
        return observation.smoothed if observation is not None else None

    def cycle_selected_landmark(self, step: int) -> LandmarkTarget:
        """Change which landmark is displayed without re-running detection.

        Tracking continues for every valid keypoint. Only the overlay target
        changes.

        Args:
            step: +1 for next catalog entry, -1 for previous.

        Returns:
            The newly selected landmark.
        """
        self.selected_landmark = cycle_landmark(self.selected_landmark, step)
        return self.selected_landmark

    def reset_tracking(self) -> None:
        """Reset smoother and motion totals after a video restart."""
        self.pose_estimator.reset_tracking()
        self.hands_estimator.reset_tracking()
        self.motion_tracker.reset()
        self._frame_index = 0

    def close(self) -> None:
        """Release MediaPipe Pose and Hands."""
        self.pose_estimator.close()
        self.hands_estimator.close()
        self.motion_tracker.reset()
