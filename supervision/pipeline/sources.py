from __future__ import annotations

from collections.abc import Generator
from typing import Any, Callable

import numpy as np

from supervision.pipeline.core import PipelineSource
from supervision.utils.capture import (
    FileVideoCapture,
    StreamCapture,
    WebcamVideoCapture,
)


class VideoFileSource(PipelineSource):
    """
    Pipeline source for video files using threaded capture.

    Uses FileVideoCapture for optimal performance with video files.

    Examples:
        ```python
        import supervision as sv

        source = sv.VideoFileSource("video.mp4")
        pipeline = sv.Pipeline(source) | sv.DisplaySink("Video")
        pipeline.run()
        ```
    """

    def __init__(
        self,
        video_path: str,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        queue_size: int = 128,
    ):
        """
        Initialize video file source.

        Args:
            video_path: Path to video file
            transform: Optional frame transformation function
            queue_size: Size of frame buffer queue
        """
        self.video_path = video_path
        self.capture = FileVideoCapture(
            src=video_path, transform=transform, queue_size=queue_size
        )
        self.frame_number = 0

    def __iter__(self) -> Generator[dict[str, Any], None, None]:
        """
        Iterate over video frames.

        Yields:
            Dictionary with 'frame', 'frame_number', and metadata
        """
        self.capture.start()
        self.frame_number = 0

        try:
            while self.capture.running():
                if self.capture.more():
                    frame = self.capture.read()
                    yield {
                        "frame": frame,
                        "frame_number": self.frame_number,
                        "source": "file",
                        "source_path": self.video_path,
                    }
                    self.frame_number += 1
        finally:
            self.capture.stop()

    def close(self) -> None:
        """Close video file capture."""
        if hasattr(self, "capture"):
            self.capture.stop()


class WebcamSource(PipelineSource):
    """
    Pipeline source for webcam capture using threaded capture.

    Uses WebcamVideoCapture for reliable webcam access with
    auto-reconnection.

    Examples:
        ```python
        import supervision as sv

        # Default webcam
        source = sv.WebcamSource()

        # Specific camera with resolution
        source = sv.WebcamSource(camera_id=1, width=1920, height=1080)

        pipeline = sv.Pipeline(source) | sv.DisplaySink("Webcam")
        pipeline.run()
        ```
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        queue_size: int = 128,
    ):
        """
        Initialize webcam source.

        Args:
            camera_id: Camera device ID (0 for default)
            width: Desired frame width
            height: Desired frame height
            fps: Desired frames per second
            transform: Optional frame transformation function
            queue_size: Size of frame buffer queue
        """
        self.camera_id = camera_id
        self.capture = WebcamVideoCapture(
            src=camera_id,
            width=width,
            height=height,
            fps=fps,
            transform=transform,
            queue_size=queue_size,
            name=f"WebcamSource_{camera_id}",
        )
        self.frame_number = 0

    def __iter__(self) -> Generator[dict[str, Any], None, None]:
        """
        Iterate over webcam frames.

        Yields:
            Dictionary with 'frame', 'frame_number', and metadata
        """
        self.capture.start()
        self.frame_number = 0

        try:
            while self.capture.running():
                if self.capture.more():
                    frame = self.capture.read()
                    health = self.capture.get_health()

                    yield {
                        "frame": frame,
                        "frame_number": self.frame_number,
                        "source": "webcam",
                        "camera_id": self.camera_id,
                        "health": health,
                    }
                    self.frame_number += 1
        finally:
            self.capture.stop()

    def close(self) -> None:
        """Close webcam capture."""
        if hasattr(self, "capture"):
            self.capture.stop()


class StreamSource(PipelineSource):
    """
    Pipeline source for network video streams (RTSP/RTMP/HTTP).

    Uses StreamCapture for robust network stream handling with
    automatic reconnection.

    Examples:
        ```python
        import supervision as sv

        # RTSP stream
        source = sv.StreamSource("rtsp://192.168.1.100:554/stream")

        # With TCP transport
        source = sv.StreamSource(
            "rtsp://camera.local/stream",
            transport="tcp",
            buffer_size=1
        )

        pipeline = sv.Pipeline(source) | sv.DisplaySink("Stream")
        pipeline.run()
        ```
    """

    def __init__(
        self,
        stream_url: str,
        transport: str = "tcp",
        use_hw_accel: bool = False,
        buffer_size: int = 1,
        queue_size: int = 128,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        """
        Initialize stream source.

        Args:
            stream_url: URL of the stream (rtsp://, rtmp://, http://)
            transport: Transport protocol for RTSP ("tcp" or "udp")
            use_hw_accel: Whether to use hardware acceleration
            buffer_size: OpenCV buffer size (lower = less latency)
            queue_size: Size of frame buffer queue
            transform: Optional frame transformation function
        """
        self.stream_url = stream_url
        self.capture = StreamCapture(
            stream_url=stream_url,
            transport=transport,
            use_hw_accel=use_hw_accel,
            buffer_size=buffer_size,
            queue_size=queue_size,
            transform=transform,
            name="StreamSource",
        )
        self.frame_number = 0

    def __iter__(self) -> Generator[dict[str, Any], None, None]:
        """
        Iterate over stream frames.

        Yields:
            Dictionary with 'frame', 'frame_number', and metadata
        """
        self.capture.start()
        self.frame_number = 0

        try:
            while self.capture.running():
                if self.capture.more():
                    frame = self.capture.read()
                    health = self.capture.get_health()

                    yield {
                        "frame": frame,
                        "frame_number": self.frame_number,
                        "source": "stream",
                        "stream_url": self.stream_url,
                        "health": health,
                    }
                    self.frame_number += 1
        finally:
            self.capture.stop()

    def close(self) -> None:
        """Close stream capture."""
        if hasattr(self, "capture"):
            self.capture.stop()


class FrameGeneratorSource(PipelineSource):
    """
    Pipeline source from a frame generator function.

    Wraps any generator that yields frames into a pipeline source.

    Examples:
        ```python
        import supervision as sv

        def my_frames():
            for i in range(100):
                yield np.zeros((480, 640, 3), dtype=np.uint8)

        source = sv.FrameGeneratorSource(my_frames())
        pipeline = sv.Pipeline(source) | sv.DisplaySink("Frames")
        pipeline.run()
        ```
    """

    def __init__(self, frame_generator: Generator[np.ndarray, None, None]):
        """
        Initialize from frame generator.

        Args:
            frame_generator: Generator yielding frames
        """
        self.frame_generator = frame_generator
        self.frame_number = 0

    def __iter__(self) -> Generator[dict[str, Any], None, None]:
        """
        Iterate over generator frames.

        Yields:
            Dictionary with 'frame', 'frame_number', and metadata
        """
        self.frame_number = 0

        for frame in self.frame_generator:
            yield {
                "frame": frame,
                "frame_number": self.frame_number,
                "source": "generator",
            }
            self.frame_number += 1

    def close(self) -> None:
        """Close generator source."""
        # Generators don't need explicit cleanup
        pass


class OakRgbSource(PipelineSource):
    """
    Pipeline source for Luxonis OAK RGB cameras using DepthAI.

    Uses OakRgbCapture for threaded frame capture from OAK-1 or the RGB
    sensor on OAK-D/OAK-D Lite/OAK-D Pro cameras.

    Examples:
        ```python
        import supervision as sv

        # Basic usage with auto-detection
        source = sv.OakRgbSource()
        pipeline = sv.Pipeline(source) | sv.DisplaySink("OAK RGB")
        pipeline.run()

        # With custom resolution and FPS
        source = sv.OakRgbSource(width=1920, height=1080, fps=30)

        # With specific device
        source = sv.OakRgbSource(device_mxid="14442C10D13EAFD000")

        # With manual exposure control
        source = sv.OakRgbSource(
            manual_exposure=10000,  # 10ms
            manual_iso=400
        )

        pipeline = sv.Pipeline(source) | sv.DisplaySink("OAK")
        pipeline.run()
        ```
    """

    def __init__(
        self,
        device_mxid: str | None = None,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        color_order: str = "BGR",
        interleaved: bool = True,
        manual_exposure: int | None = None,
        manual_iso: int | None = None,
        manual_focus: int | None = None,
        manual_white_balance: int | None = None,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        queue_size: int = 128,
    ):
        """
        Initialize OAK RGB source.

        Args:
            device_mxid: Device MxId for specific device (None for auto-detection)
            width: Frame width in pixels (max 3840)
            height: Frame height in pixels (max 2160)
            fps: Frames per second
            color_order: Color order "RGB" or "BGR" (default: "BGR")
            interleaved: True for interleaved, False for planar
            manual_exposure: Manual exposure in microseconds (1-33000, None for auto)
            manual_iso: Manual ISO (100-1600, None for auto)
            manual_focus: Manual focus (0-255, None for auto)
            manual_white_balance: Manual white balance in Kelvin (1000-12000,
                None for auto)
            transform: Optional frame transformation function
            queue_size: Size of frame buffer queue
        """
        from supervision.utils.capture import OakRgbCapture

        self.device_mxid = device_mxid
        self.capture = OakRgbCapture(
            device_mxid=device_mxid,
            width=width,
            height=height,
            fps=fps,
            color_order=color_order,
            interleaved=interleaved,
            manual_exposure=manual_exposure,
            manual_iso=manual_iso,
            manual_focus=manual_focus,
            manual_white_balance=manual_white_balance,
            transform=transform,
            queue_size=queue_size,
            name="OakRgbSource",
        )
        self.frame_number = 0

    def __iter__(self) -> Generator[dict[str, Any], None, None]:
        """
        Iterate over OAK RGB frames.

        Yields:
            Dictionary with 'frame', 'frame_number', and metadata
        """
        self.capture.start()
        self.frame_number = 0

        try:
            while self.capture.running():
                if self.capture.more():
                    frame = self.capture.read()
                    health = self.capture.get_health()
                    device_info = self.capture.get_device_info()

                    yield {
                        "frame": frame,
                        "frame_number": self.frame_number,
                        "source": "oak_rgb",
                        "device_info": device_info,
                        "health": health,
                    }
                    self.frame_number += 1
        finally:
            self.capture.stop()

    def close(self) -> None:
        """Close OAK RGB capture."""
        if hasattr(self, "capture"):
            self.capture.stop()


class OakStereoSource(PipelineSource):
    """
    Pipeline source for Luxonis OAK stereo cameras using DepthAI.

    Uses OakStereoCapture for threaded frame capture from OAK-D, OAK-D Lite,
    or OAK-D Pro cameras. Supports multiple simultaneous streams including
    left/right mono, depth, disparity, and rectified views.

    Examples:
        ```python
        import supervision as sv

        # Capture depth only (default)
        source = sv.OakStereoSource()
        pipeline = sv.Pipeline(source) | sv.DisplaySink("Depth")
        pipeline.run()

        # Capture multiple streams
        source = sv.OakStereoSource(
            output_streams=["left", "right", "depth"],
            resolution="800p"
        )

        # Access streams in pipeline
        pipeline = sv.Pipeline(source)
        for data in pipeline:
            left = data["streams"]["left"]
            right = data["streams"]["right"]
            depth = data["streams"]["depth"]
            # Process...

        # With stereo parameters
        source = sv.OakStereoSource(
            output_streams=["depth"],
            extended_disparity=True,
            subpixel=True,
            lr_check=True
        )
        ```
    """

    def __init__(
        self,
        device_mxid: str | None = None,
        resolution: str = "400p",
        output_streams: list[str] | None = None,
        extended_disparity: bool = False,
        subpixel: bool = False,
        lr_check: bool = True,
        median_filter: str = "KERNEL_7x7",
        transform: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]
        | None = None,
        queue_size: int = 128,
    ):
        """
        Initialize OAK stereo source.

        Args:
            device_mxid: Device MxId for specific device (None for auto-detection)
            resolution: Mono camera resolution ("400p", "480p", "720p", "800p")
            output_streams: List of streams to capture
                ["left", "right", "depth", "disparity", "rectified_left",
                "rectified_right"]
                Default: ["depth"]
            extended_disparity: Enable extended disparity for closer depth range
            subpixel: Enable subpixel mode for better precision
            lr_check: Enable left-right check for better occlusion handling
            median_filter: Median filter kernel ("KERNEL_3x3", "KERNEL_5x5",
                "KERNEL_7x7")
            transform: Optional transformation function for frames dict
            queue_size: Size of frame buffer queue
        """
        from supervision.utils.capture import OakStereoCapture

        self.device_mxid = device_mxid
        self.output_streams = output_streams or ["depth"]
        self.capture = OakStereoCapture(
            device_mxid=device_mxid,
            resolution=resolution,
            output_streams=self.output_streams,
            extended_disparity=extended_disparity,
            subpixel=subpixel,
            lr_check=lr_check,
            median_filter=median_filter,
            transform=transform,
            queue_size=queue_size,
            name="OakStereoSource",
        )
        self.frame_number = 0

    def __iter__(self) -> Generator[dict[str, Any], None, None]:
        """
        Iterate over OAK stereo frames.

        Yields:
            Dictionary with:
                - frame: First stream (for compatibility)
                - streams: Dict of all streams
                - frame_number: Frame counter
                - source: "oak_stereo"
                - device_info: Device information
                - health: Health status
        """
        self.capture.start()
        self.frame_number = 0

        try:
            while self.capture.running():
                if self.capture.more():
                    streams = self.capture.read()
                    health = self.capture.get_health()
                    device_info = self.capture.get_device_info()

                    default_frame = streams[self.output_streams[0]]

                    yield {
                        "frame": default_frame,
                        "streams": streams,
                        "frame_number": self.frame_number,
                        "source": "oak_stereo",
                        "device_info": device_info,
                        "health": health,
                    }
                    self.frame_number += 1
        finally:
            self.capture.stop()

    def close(self) -> None:
        """Close OAK stereo capture."""
        if hasattr(self, "capture"):
            self.capture.stop()


class OakRgbDepthSource(PipelineSource):
    """
    Pipeline source for Luxonis OAK cameras with aligned RGB + Depth.

    Uses OakRgbDepthCapture for threaded capture of aligned RGB and depth
    frames from OAK-D cameras. This is optimized for the common use case
    of needing color image with corresponding depth information.

    Examples:
        ```python
        import supervision as sv

        # Basic usage
        source = sv.OakRgbDepthSource()

        # Access RGB and depth in pipeline
        pipeline = sv.Pipeline(source)
        for data in pipeline:
            rgb = data["frame"]  # RGB frame (default)
            depth = data["depth"]  # Depth map
            # Process...

        # With custom settings
        source = sv.OakRgbDepthSource(
            rgb_width=1920,
            rgb_height=1080,
            depth_resolution="800p",
            fps=30,
            extended_disparity=True
        )

        # Display both streams
        pipeline = sv.Pipeline(source) | sv.CallbackStep(
            lambda data: {
                **data,
                "display": np.hstack([data["frame"], data["depth"]])
            }
        ) | sv.DisplaySink("RGB + Depth")
        pipeline.run()
        ```
    """

    def __init__(
        self,
        device_mxid: str | None = None,
        rgb_width: int = 1920,
        rgb_height: int = 1080,
        depth_resolution: str = "400p",
        fps: int = 30,
        color_order: str = "BGR",
        extended_disparity: bool = False,
        subpixel: bool = False,
        lr_check: bool = True,
        align_to_rgb: bool = True,
        manual_exposure: int | None = None,
        manual_iso: int | None = None,
        manual_focus: int | None = None,
        manual_white_balance: int | None = None,
        transform: Callable[
            [tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]
        ]
        | None = None,
        queue_size: int = 128,
    ):
        """
        Initialize OAK RGB + Depth source.

        Args:
            device_mxid: Device MxId (None for auto-detection)
            rgb_width: RGB frame width (max 3840)
            rgb_height: RGB frame height (max 2160)
            depth_resolution: Depth resolution ("400p", "480p", "720p", "800p")
            fps: Frames per second
            color_order: "RGB" or "BGR"
            extended_disparity: Enable extended disparity
            subpixel: Enable subpixel mode
            lr_check: Enable left-right check
            align_to_rgb: Align depth to RGB perspective
            manual_exposure: Manual exposure in μs (None for auto)
            manual_iso: Manual ISO (None for auto)
            manual_focus: Manual focus (None for auto)
            manual_white_balance: Manual white balance in K (None for auto)
            transform: Optional transformation for (rgb, depth) tuple
            queue_size: Frame buffer size
        """
        from supervision.utils.capture import OakRgbDepthCapture

        self.device_mxid = device_mxid
        self.capture = OakRgbDepthCapture(
            device_mxid=device_mxid,
            rgb_width=rgb_width,
            rgb_height=rgb_height,
            depth_resolution=depth_resolution,
            fps=fps,
            color_order=color_order,
            extended_disparity=extended_disparity,
            subpixel=subpixel,
            lr_check=lr_check,
            align_to_rgb=align_to_rgb,
            manual_exposure=manual_exposure,
            manual_iso=manual_iso,
            manual_focus=manual_focus,
            manual_white_balance=manual_white_balance,
            transform=transform,
            queue_size=queue_size,
            name="OakRgbDepthSource",
        )
        self.frame_number = 0

    def __iter__(self) -> Generator[dict[str, Any], None, None]:
        """
        Iterate over OAK RGB + Depth frames.

        Yields:
            Dictionary with:
                - frame: RGB frame
                - depth: Depth map
                - frame_number: Frame counter
                - source: "oak_rgb_depth"
                - device_info: Device information
                - health: Health status
        """
        self.capture.start()
        self.frame_number = 0

        try:
            while self.capture.running():
                if self.capture.more():
                    rgb, depth = self.capture.read()
                    health = self.capture.get_health()
                    device_info = self.capture.get_device_info()

                    yield {
                        "frame": rgb,
                        "depth": depth,
                        "frame_number": self.frame_number,
                        "source": "oak_rgb_depth",
                        "device_info": device_info,
                        "health": health,
                    }
                    self.frame_number += 1
        finally:
            self.capture.stop()

    def close(self) -> None:
        """Close OAK RGB + Depth capture."""
        if hasattr(self, "capture"):
            self.capture.stop()
