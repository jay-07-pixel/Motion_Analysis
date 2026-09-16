"""Application entry point for live camera and uploaded-video 2D analysis."""

from __future__ import annotations

import sys
from time import perf_counter
from typing import Optional

from motion_analysis.camera import (
    CameraError,
    FrameSource,
    RealSenseColorCamera,
    VideoFileSource,
)
from motion_analysis.display import (
    LiveRgbViewer,
    ViewerAction,
    show_input_menu,
    show_landmark_picker,
)
from motion_analysis.display.menu import InputChoice
from motion_analysis.hands import HandsInitializationError
from motion_analysis.motion import LandmarkTarget, default_landmark, draw_motion_analysis
from motion_analysis.pipeline import FrameProcessor
from motion_analysis.pipeline.processor import ProcessedFrame
from motion_analysis.pose import PoseInitializationError


def run_session(
    source: FrameSource,
    is_video: bool,
    selected_landmark: LandmarkTarget,
) -> int:
    """Open one frame source and run the shared 2D pipeline until exit.

    Args:
        source: Live RealSense camera or uploaded video file.
        is_video: True when the source is a video file (pause/restart keys).
        selected_landmark: Body or hand joint to show motion measurements for.

    Returns:
        0 on a clean quit, 1 if the source or models cannot start.
    """
    viewer = LiveRgbViewer()
    processor: Optional[FrameProcessor] = None
    paused = False
    last_result: Optional[ProcessedFrame] = None
    try:
        metadata = source.open()
        print(f"Source: {metadata.source_name}")
        print(f"Resolution: {metadata.width} x {metadata.height}")
        fps_text = f"{metadata.stream_fps:.1f}" if metadata.stream_fps > 0 else "unknown"
        print(f"Source FPS: {fps_text}")

        processor = FrameProcessor(selected_landmark=selected_landmark)
        print("MediaPipe Pose and Hands ready. Both inputs share one 2D pipeline.")
        print(f"Analysing: {processor.selected_landmark.label}")
        print("Press [ or ] in the video window to change landmark.")
        if is_video:
            print("Video controls: Space pause/play, R restart, Q or Esc exit.")
        else:
            print("Press Q or Esc in the video window to quit.")

        wait_ms = _playback_delay_ms(metadata.stream_fps) if is_video else 1

        while True:
            if not paused:
                frame = source.read()
                if frame is None:
                    paused = True
                    if last_result is None:
                        print("The video file produced no frames.", file=sys.stderr)
                        return 1
                else:
                    last_result = processor.process(
                        frame,
                        timestamp_seconds=_frame_timestamp(source),
                        source_fps=source.get_metadata().stream_fps,
                    )

            if last_result is None:
                continue

            display_frame = draw_motion_analysis(
                last_result.annotated,
                processor.current_measurement(),
                selected_keypoint=processor.selected_keypoint(last_result),
                selected_raw=_selected_raw(processor, last_result),
            )
            extra_lines = _overlay_lines(source, is_video, paused)
            action = viewer.show(
                display_frame,
                source.get_metadata(),
                extra_lines=extra_lines,
                wait_ms=50 if paused else wait_ms,
            )
            if action is ViewerAction.QUIT:
                break
            if action is ViewerAction.TOGGLE_PAUSE and is_video:
                paused = not paused
            if action is ViewerAction.RESTART and is_video:
                source.restart()
                processor.reset_tracking()
                paused = False
                last_result = None
            if action is ViewerAction.PREV_LANDMARK:
                processor.cycle_selected_landmark(-1)
                print(f"Analysing: {processor.selected_landmark.label}")
            if action is ViewerAction.NEXT_LANDMARK:
                processor.cycle_selected_landmark(1)
                print(f"Analysing: {processor.selected_landmark.label}")
        return 0
    except CameraError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 1
    except PoseInitializationError as exc:
        print(f"Pose error: {exc}", file=sys.stderr)
        return 1
    except HandsInitializationError as exc:
        print(f"Hands error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 0
    finally:
        if processor is not None:
            processor.close()
        viewer.close()
        source.close()


def _selected_raw(processor: FrameProcessor, result: ProcessedFrame):
    """Return the selected landmark's raw keypoint when it exists.

    Args:
        processor: Active frame processor with the current landmark target.
        result: Latest pipeline output.

    Returns:
        Raw PixelKeypoint, or None if that joint was not detected.
    """
    observation = processor.selected_observation(result)
    return observation.raw if observation is not None else None


def _frame_timestamp(source: FrameSource) -> float:
    """Read the actual timestamp for the frame just returned by ``read()``.

    Args:
        source: Live camera or uploaded video.

    Returns:
        Source timestamp in seconds. Falls back to a monotonic clock when
        the source has no time, instead of assuming a hard-coded FPS.
    """
    timestamp = source.last_frame_time_seconds()
    if timestamp is None:
        return perf_counter()
    return timestamp


def _playback_delay_ms(source_fps: float) -> int:
    """Convert a video's reported FPS into an OpenCV wait delay.

    Args:
        source_fps: Frames per second read from the file. 0 means unknown.

    Returns:
        Delay in milliseconds. Unknown FPS uses 1 ms so playback is not
        forced to a hard-coded rate.
    """
    if source_fps <= 0:
        return 1
    return max(1, int(round(1000.0 / source_fps)))


def _overlay_lines(source: FrameSource, is_video: bool, paused: bool) -> list[str]:
    """Build source-specific overlay rows.

    Args:
        source: Active camera or video source.
        is_video: True for uploaded video playback.
        paused: Whether video playback is paused.

    Returns:
        Extra overlay lines for the OpenCV window.
    """
    if not is_video:
        return ["Live camera", "Press Q or Esc to quit"]

    video = source
    assert isinstance(video, VideoFileSource)
    status = video.playback_status()
    frame_index = int(status["frame_index"])
    frame_count = int(status["frame_count"])
    position = float(status["position_seconds"])
    duration = float(status["duration_seconds"])
    ended = bool(status["ended"])
    frame_text = (
        f"Frame {frame_index}/{frame_count}" if frame_count > 0 else f"Frame {frame_index}"
    )
    time_text = f"Time {position:.2f}s"
    if duration > 0:
        time_text += f" / {duration:.2f}s"
    state = "paused" if paused else "playing"
    if ended:
        state = "ended"
    return [
        f"{frame_text}   {time_text}   {state}",
        "Space pause/play   R restart   Q/Esc exit",
    ]


def _source_from_choice(choice: InputChoice) -> FrameSource:
    """Create the frame source for a menu selection.

    Args:
        choice: Live camera or a picker-selected video path.

    Returns:
        A FrameSource that feeds the shared 2D processor.

    Raises:
        CameraError: If the video path is missing.
    """
    if choice.mode == "camera":
        return RealSenseColorCamera()
    if not choice.video_path:
        raise CameraError("No video file was selected.")
    return VideoFileSource(choice.video_path)


def main() -> int:
    """Show the input and landmark menus, then run live camera or video.

    Returns:
        0 when the user leaves the menu, otherwise the last session code.
    """
    last_code = 0
    last_landmark = default_landmark()
    while True:
        try:
            choice = show_input_menu()
        except RuntimeError as exc:
            print(f"Menu error: {exc}", file=sys.stderr)
            return 1
        if choice is None:
            return last_code
        try:
            selected = show_landmark_picker(last_landmark)
        except RuntimeError as exc:
            print(f"Menu error: {exc}", file=sys.stderr)
            return 1
        if selected is None:
            continue
        last_landmark = selected
        try:
            source = _source_from_choice(choice)
        except CameraError as exc:
            print(f"Input error: {exc}", file=sys.stderr)
            last_code = 1
            continue
        last_code = run_session(
            source,
            is_video=choice.mode == "video",
            selected_landmark=selected,
        )
