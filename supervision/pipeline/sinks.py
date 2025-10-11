from __future__ import annotations

from typing import Any, Callable

import cv2

from supervision.pipeline.core import PipelineSink
from supervision.utils.video import VideoInfo
from supervision.utils.video import VideoSink as SVVideoSink


class DisplaySink(PipelineSink):
    """
    Pipeline sink to display frames in a window.

    Displays frames using cv2.imshow with keyboard controls.

    Examples:
        ```python
        import supervision as sv

        pipeline = (
            sv.Pipeline(source)
            | sv.FPSCalculatorStep()
            | sv.DisplaySink("Video", show_fps=True)
        )
        pipeline.run()
        ```
    """

    def __init__(
        self,
        window_name: str = "Pipeline",
        show_fps: bool = False,
        wait_key: int = 1,
        window_mode: int = cv2.WINDOW_AUTOSIZE,
    ):
        """
        Initialize display sink.

        Args:
            window_name: Name of the display window
            show_fps: Whether to show FPS on the frame
            wait_key: Delay in milliseconds for cv2.waitKey (1 = real-time)
            window_mode: Window mode (cv2.WINDOW_AUTOSIZE or cv2.WINDOW_NORMAL)
        """
        self.window_name = window_name
        self.show_fps = show_fps
        self.wait_key = wait_key
        self.window_mode = window_mode
        self.stopped = False

        cv2.namedWindow(self.window_name, self.window_mode)

    def consume(self, data: dict[str, Any]) -> bool | None:
        """
        Display frame.

        Args:
            data: Pipeline data containing 'frame'

        Returns:
            False to stop, True to continue
        """
        if self.stopped:
            return False

        frame = data.get("frame")
        if frame is None:
            return

        # Add FPS overlay if requested
        if self.show_fps and "fps" in data:
            fps = data["fps"]
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

        cv2.imshow(self.window_name, frame)

        # Check for quit keys
        key = cv2.waitKey(self.wait_key) & 0xFF
        if key == 27 or key == ord("q"):  # ESC or 'q'
            self.stopped = True
            return False  # Signal to stop
        return True  # Continue

    def close(self) -> None:
        """Close display window."""
        cv2.destroyWindow(self.window_name)


class VideoFileSink(PipelineSink):
    """
    Pipeline sink to save frames to a video file.

    Examples:
        ```python
        import supervision as sv

        pipeline = (
            sv.Pipeline(source)
            | sv.AnnotationStep(annotator)
            | sv.VideoFileSink("output.mp4", fps=30, width=1920, height=1080)
        )
        pipeline.run()
        ```
    """

    def __init__(
        self,
        output_path: str,
        fps: int = 30,
        width: int = 1920,
        height: int = 1080,
        codec: str = "mp4v",
    ):
        """
        Initialize video file sink.

        Args:
            output_path: Path to output video file
            fps: Frames per second
            width: Video width
            height: Video height
            codec: Video codec (FOURCC code)
        """
        self.output_path = output_path
        video_info = VideoInfo(width=width, height=height, fps=fps)
        self.video_sink = SVVideoSink(
            target_path=output_path, video_info=video_info, codec=codec
        )
        self.video_sink.__enter__()

    def consume(self, data: dict[str, Any]) -> bool | None:
        """
        Write frame to video file.

        Args:
            data: Pipeline data containing 'frame'

        Returns:
            None to continue
        """
        frame = data.get("frame")
        if frame is not None:
            self.video_sink.write_frame(frame)
        return None

    def close(self) -> None:
        """Close video file."""
        self.video_sink.__exit__(None, None, None)


class CallbackSink(PipelineSink):
    """
    Pipeline sink that calls a custom function for each frame.

    Useful for custom output handling or side effects.

    Examples:
        ```python
        import supervision as sv

        def save_detections(data):
            frame_num = data['frame_number']
            detections = data.get('detections')
            if detections:
                print(f"Frame {frame_num}: {len(detections)} detections")

        pipeline = (
            sv.Pipeline(source)
            | sv.DetectionStep(model)
            | sv.CallbackSink(save_detections)
        )
        pipeline.run()
        ```
    """

    def __init__(self, callback: Callable[[dict[str, Any]], None]):
        """
        Initialize callback sink.

        Args:
            callback: Function to call with each data dictionary
        """
        self.callback = callback

    def consume(self, data: dict[str, Any]) -> bool | None:
        """
        Execute callback with data.

        Returns:
            None to continue
        """
        self.callback(data)
        return None

    def close(self) -> None:
        """No cleanup needed."""
        pass


class MultiSink(PipelineSink):
    """
    Pipeline sink that forwards data to multiple sinks.

    Allows sending output to multiple destinations simultaneously.

    Examples:
        ```python
        import supervision as sv

        sink = sv.MultiSink([
            sv.DisplaySink("Live"),
            sv.VideoFileSink("output.mp4", fps=30, width=1920, height=1080),
        ])

        pipeline = sv.Pipeline(source) | sv.AnnotationStep(annotator) | sink
        pipeline.run()
        ```
    """

    def __init__(self, sinks: list[PipelineSink]):
        """
        Initialize multi-sink.

        Args:
            sinks: List of sinks to forward data to
        """
        self.sinks = sinks

    def consume(self, data: dict[str, Any]) -> bool | None:
        """
        Forward data to all sinks.

        Returns:
            False if any sink returns False, None otherwise
        """
        for sink in self.sinks:
            result = sink.consume(data)
            if result is False:
                return False
        return None

    def close(self) -> None:
        """Close all sinks."""
        for sink in self.sinks:
            sink.close()
