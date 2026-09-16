"""Discovery of connected Intel RealSense devices with a color stream."""

from __future__ import annotations

from dataclasses import dataclass

from motion_analysis.camera.exceptions import CameraNotFoundError

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover - depends on local SDK install
    rs = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class RealSenseDeviceInfo:
    """Identifying details for a connected RealSense device.

    Attributes:
        name: Product name reported by the firmware, for example D455.
        serial_number: Unique device serial used to open the camera.
        firmware_version: Firmware version string from the device.
        usb_type_descriptor: USB connection type, if the SDK reports it.
        has_color_stream: True when at least one RGB/color profile exists.
    """

    name: str
    serial_number: str
    firmware_version: str
    usb_type_descriptor: str
    has_color_stream: bool


def _require_sdk() -> None:
    """Raise a clear error if pyrealsense2 is not importable."""
    if rs is None:
        raise CameraNotFoundError(
            "The Intel RealSense SDK Python package (pyrealsense2) is not "
            "installed or could not be imported. Install project requirements "
            "and the Intel RealSense SDK, then reconnect the camera. "
            f"Original error: {_IMPORT_ERROR}"
        )


def _read_info(device: rs.device, info: rs.camera_info, fallback: str = "unknown") -> str:
    """Read a camera_info field when the device supports it.

    Args:
        device: RealSense device handle from the SDK context.
        info: SDK info enum to query.
        fallback: Value returned when the field is unavailable.

    Returns:
        The info string, or fallback if the device does not expose it.
    """
    if device.supports(info):
        value = device.get_info(info)
        return value if value else fallback
    return fallback


def device_has_color_stream(device: rs.device) -> bool:
    """Return whether a RealSense device can produce an RGB/color stream.

    Args:
        device: Connected RealSense device.

    Returns:
        True if any sensor advertises a color video profile.
    """
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() == rs.stream.color:
                return True
    return False


def list_realsense_devices() -> list[RealSenseDeviceInfo]:
    """Query the RealSense SDK for connected cameras.

    Returns:
        One entry per connected RealSense device. The list may include
        devices without a color stream; callers should filter as needed.

    Raises:
        CameraNotFoundError: If the SDK is missing.
    """
    _require_sdk()
    context = rs.context()
    devices: list[RealSenseDeviceInfo] = []
    for device in context.query_devices():
        devices.append(
            RealSenseDeviceInfo(
                name=_read_info(device, rs.camera_info.name),
                serial_number=_read_info(device, rs.camera_info.serial_number),
                firmware_version=_read_info(device, rs.camera_info.firmware_version),
                usb_type_descriptor=_read_info(device, rs.camera_info.usb_type_descriptor),
                has_color_stream=device_has_color_stream(device),
            )
        )
    return devices


def find_color_camera() -> RealSenseDeviceInfo:
    """Select a connected RealSense camera that supports RGB capture.

    If several color-capable devices are attached, a D455-family camera is
    preferred because this project targets the Intel RealSense D455f.
    Otherwise the first color-capable device is used.

    Returns:
        Device info for the selected camera.

    Raises:
        CameraNotFoundError: If no RealSense camera with a color stream is found.
    """
    devices = list_realsense_devices()
    color_devices = [device for device in devices if device.has_color_stream]
    if not color_devices:
        connected = ", ".join(device.name for device in devices) or "none"
        raise CameraNotFoundError(
            "No Intel RealSense camera with an RGB/color stream was detected. "
            "Connect a D455f (or another color-capable RealSense camera) with "
            "a USB 3 cable, then retry. "
            f"Devices currently visible to the SDK: {connected}."
        )

    preferred = [device for device in color_devices if "D455" in device.name.upper()]
    return preferred[0] if preferred else color_devices[0]
