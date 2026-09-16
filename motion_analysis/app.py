"""Application entry point for live RealSense RGB capture and 2D pose."""

from __future__ import annotations

import sys

from motion_analysis.camera import (
    CameraError,
    RealSenseColorCamera,
)
from motion_analysis.display import LiveRgbViewer
from motion_analysis.pose import (
    PoseDetectionError,
    PoseEstimator,
    PoseInitializationError,
    draw_pose_on_frame,
)


def run_live_rgb() -> int:
    """Detect the RealSense camera, overlay 2D pose, and release resources.

    Returns:
        Process exit code: 0 on a clean quit, 1 if the camera or pose model
        cannot be initialized.
    """
    camera = RealSenseColorCamera()
    viewer = LiveRgbViewer()
    estimator: PoseEstimator | None = None
    try:
        metadata = camera.open()
        print(f"Connected: {metadata.source_name}")
        print(f"Resolution: {metadata.width} x {metadata.height}")
        print(f"Camera FPS: {metadata.stream_fps:.1f}")

        estimator = PoseEstimator()
        print("MediaPipe Pose ready. 2D keypoints are reported in pixel coordinates.")
        print("Press Q or Esc in the video window to quit.")

        while True:
            frame = camera.read()
            pose_frame = None
            pose_error = None
            try:
                pose_frame = estimator.estimate(frame)
            except PoseDetectionError as exc:
                pose_error = str(exc)

            annotated = draw_pose_on_frame(frame, pose_frame, error_message=pose_error)
            if not viewer.show(annotated, camera.get_metadata()):
                break
        return 0
    except CameraError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1
    except PoseInitializationError as exc:
        print(f"Pose error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 0
    finally:
        if estimator is not None:
            estimator.close()
        viewer.close()
        camera.close()


def main() -> int:
    """CLI wrapper used by `python main.py` and `python -m motion_analysis`.

    Returns:
        Exit code from the live RGB pose session.
    """
    return run_live_rgb()
