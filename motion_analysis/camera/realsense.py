"""Intel RealSense RGB/color stream backend."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from motion_analysis.camera.base import FrameMetadata, FrameSource
from motion_analysis.camera.detector import RealSenseDeviceInfo, find_color_camera
from motion_analysis.camera.exceptions import (
    CameraInitializationError,
    CameraNotFoundError,
    CameraReadError,
)

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - import is validated at open()
    rs = None


class RealSenseColorCamera(FrameSource):
    """Captures live RGB frames from a connected Intel RealSense camera.

    Resolution and FPS are taken from the active RealSense stream profile
    after start-up. They are never hard-coded.

    Args:
        device_info: Optional pre-selected device. When omitted, a connected
            color-capable RealSense camera is detected automatically.
        frame_timeout_ms: Maximum time to wait for each color frame.
    """

    def __init__(
        self,
        device_info: Optional[RealSenseDeviceInfo] = None,
        frame_timeout_ms: int = 5000,
    ) -> None:
        self._requested_device = device_info
        self._frame_timeout_ms = frame_timeout_ms
        self._context: Optional[rs.context] = None
        self._pipeline: Optional[rs.pipeline] = None
        self._device_info: Optional[RealSenseDeviceInfo] = None
        self._metadata: Optional[FrameMetadata] = None
        self._color_format: Optional[rs.format] = None

    def open(self) -> FrameMetadata:
        """Detect the camera if needed and start the RGB/color stream.

        Returns:
            Metadata with the actual stream resolution and FPS.

        Raises:
            CameraNotFoundError: If no compatible RealSense camera is connected.
            CameraInitializationError: If the color stream cannot be started.
        """
        if rs is None:
            raise CameraNotFoundError(
                "pyrealsense2 is not installed. Install project requirements "
                "and the Intel RealSense SDK before running the live camera."
            )

        self.close()
        self._device_info = self._requested_device or find_color_camera()
        self._context = rs.context()
        device = _device_by_serial(self._context, self._device_info.serial_number)
        if device is None:
            raise CameraNotFoundError(
                f"RealSense device {self._device_info.serial_number} was detected "
                "but disappeared before the stream could start. Reconnect the "
                "camera and try again."
            )

        pipeline = rs.pipeline(self._context)
        try:
            profile = _start_color_pipeline(pipeline, device)
        except RuntimeError as exc:
            self._context = None
            raise CameraInitializationError(
                "Failed to initialize the RealSense RGB/color stream on "
                f"{self._device_info.name} (serial {self._device_info.serial_number}). "
                "Close other apps using the camera, reconnect the USB 3 cable, "
                f"and retry. SDK error: {exc}"
            ) from exc

        self._pipeline = pipeline
        try:
            self._metadata = self._read_stream_metadata(profile)
        except Exception:
            self.close()
            raise
        return self._metadata

    def read(self) -> np.ndarray:
        """Wait for the next RGB frame and return it in OpenCV BGR format.

        Returns:
            Color frame as a uint8 array of shape (height, width, 3).

        Raises:
            CameraReadError: If the pipeline is closed or a frame times out.
        """
        if self._pipeline is None:
            raise CameraReadError("The RealSense camera is not open.")

        try:
            frames = self._pipeline.wait_for_frames(self._frame_timeout_ms)
        except RuntimeError as exc:
            raise CameraReadError(
                "Timed out waiting for a RealSense color frame. The camera may "
                f"have been disconnected. SDK error: {exc}"
            ) from exc

        color_frame = frames.get_color_frame()
        if not color_frame:
            raise CameraReadError("The RealSense pipeline returned no color frame.")

        image = np.asanyarray(color_frame.get_data())
        return _to_bgr(image, self._color_format)

    def get_metadata(self) -> FrameMetadata:
        """Return the active color stream resolution and FPS.

        Returns:
            Metadata collected from the live RealSense profile.

        Raises:
            CameraReadError: If the camera has not been opened yet.
        """
        if self._metadata is None:
            raise CameraReadError("The RealSense camera is not open.")
        return self._metadata

    def close(self) -> None:
        """Stop the pipeline and release the RealSense device."""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except RuntimeError:
                # stop() can fail if start() never completed; ignore so close
                # remains safe to call from exception handlers.
                pass
            self._pipeline = None
        self._context = None
        self._metadata = None
        self._color_format = None

    def _read_stream_metadata(self, profile: rs.pipeline_profile) -> FrameMetadata:
        """Extract actual color resolution, FPS, and pixel format.

        Args:
            profile: Active pipeline profile returned by pipeline.start().

        Returns:
            Frame metadata for overlay and later processing stages.

        Raises:
            CameraInitializationError: If no color video profile is active.
        """
        try:
            stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        except RuntimeError as exc:
            raise CameraInitializationError(
                "The RealSense pipeline started, but no RGB/color video "
                "profile is active. The connected device may not expose a "
                f"color sensor. SDK error: {exc}"
            ) from exc

        self._color_format = stream.format()
        device_name = self._device_info.name if self._device_info else "Intel RealSense"
        serial = self._device_info.serial_number if self._device_info else "unknown"
        return FrameMetadata(
            source_name=f"{device_name} ({serial})",
            width=stream.width(),
            height=stream.height(),
            stream_fps=float(stream.fps()),
        )


def _device_by_serial(context: rs.context, serial_number: str) -> Optional[rs.device]:
    """Find a connected RealSense device by serial number.

    Args:
        context: Live RealSense SDK context.
        serial_number: Serial reported during detection.

    Returns:
        Matching device handle, or None if it is no longer connected.
    """
    for device in context.query_devices():
        if device.supports(rs.camera_info.serial_number):
            if device.get_info(rs.camera_info.serial_number) == serial_number:
                return device
    return None


def _start_color_pipeline(pipeline: rs.pipeline, device: rs.device) -> rs.pipeline_profile:
    """Start a color stream using a device-provided profile.

    First lets the SDK pick a default color stream. If that fails, selects a
    color profile advertised by the connected camera.

    Args:
        pipeline: RealSense pipeline owned by this camera instance.
        device: Connected device that will provide color frames.

    Returns:
        The active pipeline profile after a successful start.

    Raises:
        RuntimeError: If no color stream can be started.
        CameraInitializationError: If the device has no color video profiles.
    """
    serial = device.get_info(rs.camera_info.serial_number)
    config = rs.config()
    config.enable_device(serial)
    # Width, height, and FPS are omitted so the SDK can choose a valid profile.
    config.enable_stream(rs.stream.color)
    try:
        return pipeline.start(config)
    except RuntimeError:
        color_profile = _select_color_profile(device)
        fallback = rs.config()
        fallback.enable_device(serial)
        fallback.enable_stream(
            rs.stream.color,
            color_profile.width(),
            color_profile.height(),
            color_profile.format(),
            color_profile.fps(),
        )
        return pipeline.start(fallback)


def _select_color_profile(device: rs.device) -> rs.video_stream_profile:
    """Choose a color video profile advertised by the connected camera.

    Args:
        device: Connected RealSense device.

    Returns:
        A color profile using values reported by the hardware.

    Raises:
        CameraInitializationError: If the device exposes no color profiles.
    """
    profiles: list[rs.video_stream_profile] = []
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() == rs.stream.color and profile.is_video_stream_profile():
                profiles.append(profile.as_video_stream_profile())
    if not profiles:
        raise CameraInitializationError(
            "The connected RealSense device does not advertise any RGB/color "
            "stream profiles."
        )

    # Prefer formats OpenCV can display, then larger frames, then higher FPS.
    format_rank = {
        rs.format.bgr8: 0,
        rs.format.rgb8: 1,
        rs.format.bgra8: 2,
        rs.format.rgba8: 3,
        rs.format.yuyv: 4,
    }

    def rank(profile: rs.video_stream_profile) -> tuple[int, int, int]:
        return (
            format_rank.get(profile.format(), 99),
            -(profile.width() * profile.height()),
            -profile.fps(),
        )

    return sorted(profiles, key=rank)[0]


def _to_bgr(image: np.ndarray, color_format: Optional[rs.format]) -> np.ndarray:
    """Convert a RealSense color frame to OpenCV BGR when needed.

    Args:
        image: Raw color image from the SDK.
        color_format: Pixel format of the active color stream.

    Returns:
        BGR image suitable for `cv2.imshow`.
    """
    if rs is None or color_format is None or color_format == rs.format.bgr8:
        return image
    if color_format == rs.format.rgb8:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if color_format == rs.format.bgra8:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if color_format == rs.format.rgba8:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if color_format == rs.format.yuyv:
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)
    return image
