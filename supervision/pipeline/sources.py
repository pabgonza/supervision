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
