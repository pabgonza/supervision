from __future__ import annotations

import time
import warnings
from queue import Queue
from threading import Thread
from typing import Callable

import cv2
import numpy as np


class WebcamVideoCapture:
    """
    Threaded video capture class for webcams and IP cameras with automatic reconnection.

    This class captures video frames in a separate thread using a queue buffer,
    providing faster and more reliable frame access compared to standard
    cv2.VideoCapture. It includes automatic reconnection, watchdog monitoring,
    and health checking.

    Attributes:
        src: Video source (0 for default webcam, URL for IP camera, etc.)
        queue: Thread-safe queue containing captured frames
        stopped: Flag indicating if capture should stop

    Examples:
        ```python
        import supervision as sv
        import cv2

        # Basic usage
        capture = sv.WebcamVideoCapture(src=0).start()
        while capture.running():
            frame = capture.read()
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.stop()
        cv2.destroyAllWindows()

        # Using context manager
        with sv.WebcamVideoCapture(src=0) as capture:
            for _ in range(100):
                if capture.more():
                    frame = capture.read()
                    # Process frame...

        # With custom resolution and FPS
        capture = sv.WebcamVideoCapture(
            src=0,
            width=1920,
            height=1080,
            fps=30
        ).start()
        ```
    """

    def __init__(
        self,
        src: int | str = 0,
        fourcc: str | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        queue_size: int = 128,
        name: str = "WebcamVideoCapture",
        use_hw_accel: bool = False,
        target_fps: int | None = None,
    ):
        """
        Initialize the WebcamVideoCapture.

        Args:
            src: Video source (0 for default webcam, camera index, or URL string)
            fourcc: Four character code for video codec (e.g., "MJPG", "YUYV")
            width: Desired frame width in pixels
            height: Desired frame height in pixels
            fps: Desired frames per second
            transform: Optional function to transform each frame before queuing
            queue_size: Maximum number of frames to buffer in queue
            name: Name for the capture instance (used in warnings)
            use_hw_accel: Whether to use hardware acceleration (FFMPEG backend)
            target_fps: If set, limits capture rate to this FPS
        """
        self.src = src
        self.use_hw_accel = use_hw_accel
        self.target_fps = target_fps
        self.queue: Queue = Queue(maxsize=queue_size)
        self.stopped = False
        self.transform = transform
        self.name = name
        self.reconnect_attempts = 0
        self.total_reconnect_attempts = 0
        self.last_frame_time = time.time()

        self.initialize_capture(fourcc, width, height, fps)

        self.thread = Thread(target=self.update, args=(), name=name)
        self.thread.daemon = True

        self.watchdog_thread = Thread(target=self.watchdog, name=f"{name}_watchdog")
        self.watchdog_thread.daemon = True

    def initialize_capture(
        self,
        fourcc: str | None,
        width: int | None,
        height: int | None,
        fps: int | None,
    ) -> None:
        """
        Initialize or reinitialize the video capture device.

        Args:
            fourcc: Four character code for video codec
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
        """
        backend = cv2.CAP_FFMPEG if self.use_hw_accel else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.src, backend)
        if not self.cap.isOpened():
            raise OSError(f"Cannot open video source {self.src}")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 30)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 60000)

        if self.use_hw_accel:
            self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            self.cap.set(cv2.CAP_PROP_HW_DEVICE, 0)

        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

    def start(self) -> WebcamVideoCapture:
        """
        Start the capture and watchdog threads.

        Returns:
            Self for method chaining.
        """
        self.thread.start()
        self.watchdog_thread.start()
        return self

    def update(self) -> None:
        """
        Main loop running in separate thread to continuously capture frames.

        This method runs until stopped flag is set. It handles reconnection
        on failures and applies optional transforms to frames.
        """
        consecutive_successful_reads = 0
        while not self.stopped:
            try:
                if not self.cap.isOpened():
                    self.reconnect()
                    continue

                if self.target_fps:
                    time.sleep(1 / self.target_fps)

                if self.queue.full():
                    self.queue.get()

                ret, frame = self.cap.read()
                if not ret:
                    warnings.warn(f"Error reading frame from {self.name}")
                    self.reconnect()
                    continue

                self.last_frame_time = time.time()

                if self.transform:
                    frame = self.transform(frame)

                self.queue.put(frame)

                consecutive_successful_reads += 1
                if consecutive_successful_reads >= 100:
                    self.reconnect_attempts = 0
                    consecutive_successful_reads = 0

            except Exception as e:
                warnings.warn(f"Unexpected error in update loop for {self.name}: {e!s}")
                self.reconnect()
                time.sleep(1)

    def reconnect(self) -> None:
        """
        Attempt to reconnect to the video source with exponential backoff.

        Uses exponential backoff up to 30 seconds between attempts.
        """
        self.total_reconnect_attempts += 1
        wait_time = min(30, 2 ** (self.total_reconnect_attempts % 10))
        warnings.warn(
            f"Attempting to reconnect {self.name}... "
            f"(Total attempts: {self.total_reconnect_attempts})"
        )
        self.cap.release()
        time.sleep(wait_time)

        try:
            self.initialize_capture(None, None, None, None)
            if self.cap.isOpened():
                warnings.warn(
                    f"Reconnection successful for {self.name} "
                    f"after {self.total_reconnect_attempts} attempts"
                )
            else:
                raise OSError("Failed to open capture")
        except Exception as e:
            warnings.warn(f"Reconnection failed for {self.name}: {e!s}")

    def watchdog(self) -> None:
        """
        Watchdog thread that monitors frame reception and triggers reconnection.

        Checks every 5 seconds if frames have been received in the last 30 seconds.
        """
        while not self.stopped:
            if time.time() - self.last_frame_time > 30:
                warnings.warn(
                    f"Watchdog detected no frames for {self.name}, restarting capture"
                )
                self.reconnect()
            time.sleep(5)

    def read(self) -> np.ndarray:
        """
        Read the next frame from the queue (blocking).

        Returns:
            The next frame as a numpy array.
        """
        return self.queue.get()

    def running(self) -> bool:
        """
        Check if capture is still running or has frames available.

        Returns:
            True if there are frames available or capture is not stopped.
        """
        return self.more() or not self.stopped

    def more(self) -> bool:
        """
        Check if there are frames available in the queue.

        Waits up to 0.5 seconds for frames to become available.

        Returns:
            True if frames are available in the queue.
        """
        tries = 0
        while self.queue.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1
        return self.queue.qsize() > 0

    def stop(self) -> None:
        """
        Stop the capture and release resources.

        Stops both capture and watchdog threads and releases the video capture.
        """
        self.stopped = True
        self.thread.join()
        self.watchdog_thread.join()
        if self.cap:
            self.cap.release()

    def get_health(self) -> dict[str, bool | float | int]:
        """
        Get health status information about the capture.

        Returns:
            Dictionary containing:
                - is_running: Whether capture is active
                - is_capture_open: Whether cv2.VideoCapture is open
                - last_frame_time: Timestamp of last successful frame
                - reconnect_attempts: Current reconnection attempts
                - queue_size: Number of frames in queue
        """
        return {
            "is_running": not self.stopped,
            "is_capture_open": self.cap.isOpened() if self.cap else False,
            "last_frame_time": self.last_frame_time,
            "reconnect_attempts": self.reconnect_attempts,
            "queue_size": self.queue.qsize(),
        }

    def get(self, prop_id: int) -> float:
        """
        Get a property value from the underlying VideoCapture.

        Args:
            prop_id: OpenCV property identifier (e.g., cv2.CAP_PROP_FRAME_WIDTH)

        Returns:
            The property value.
        """
        return self.cap.get(prop_id)

    def set(self, prop_id: int, value: float) -> bool:
        """
        Set a property value on the underlying VideoCapture.

        Args:
            prop_id: OpenCV property identifier
            value: Value to set

        Returns:
            True if successful.
        """
        return self.cap.set(prop_id, value)

    def is_alive(self) -> bool:
        """
        Check if the capture is alive and functioning.

        Returns:
            True if not stopped and capture is open.
        """
        return not self.stopped and self.cap.isOpened()

    def __enter__(self) -> WebcamVideoCapture:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


class FileVideoCapture:
    """
    Threaded video file capture class for faster frame reading.

    This class reads video files in a separate thread using a queue buffer,
    providing significant speedup compared to standard cv2.VideoCapture
    for video file processing.

    Credits: Based on https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/

    Attributes:
        cap: The underlying cv2.VideoCapture object
        queue: Thread-safe queue containing captured frames
        stopped: Flag indicating if capture should stop

    Examples:
        ```python
        import supervision as sv
        import cv2

        # Basic usage
        capture = sv.FileVideoCapture("video.mp4").start()
        while capture.running():
            frame = capture.read()
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.stop()
        cv2.destroyAllWindows()

        # Using context manager
        with sv.FileVideoCapture("video.mp4") as capture:
            while capture.running():
                frame = capture.read()
                # Process frame...
        ```
    """

    def __init__(
        self,
        src: str,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        queue_size: int = 128,
        name: str = "FileVideoCapture",
    ):
        """
        Initialize the FileVideoCapture.

        Args:
            src: Path to video file
            transform: Optional function to transform each frame before queuing
            queue_size: Maximum number of frames to buffer in queue
            name: Name for the capture instance (used in thread naming)
        """
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise OSError(f"Cannot open video file {src}")

        self.transform = transform
        self.queue: Queue = Queue(maxsize=queue_size)
        self.stopped = False

        self.thread = Thread(target=self.update, args=(), name=name)
        self.thread.daemon = True

    def start(self) -> FileVideoCapture:
        """
        Start the capture thread.

        Returns:
            Self for method chaining.
        """
        self.thread.start()
        return self

    def get(self, cv2_prop: int) -> float:
        """
        Get a property value from the underlying VideoCapture.

        Args:
            cv2_prop: OpenCV property identifier (e.g., cv2.CAP_PROP_FRAME_WIDTH)

        Returns:
            The property value.
        """
        return self.cap.get(cv2_prop)

    def update(self) -> None:
        """
        Main loop running in separate thread to continuously read frames.

        This method runs until the video ends or stopped flag is set.
        It applies optional transforms to frames before queuing.
        """
        while self.cap.isOpened():
            if self.stopped:
                break

            if not self.queue.full():
                grabbed, frame = self.cap.read()

                if not grabbed:
                    self.stopped = True
                    break

                if self.transform:
                    frame = self.transform(frame)

                self.queue.put(frame)
            else:
                time.sleep(0.01)

        self.cap.release()

    def read(self) -> np.ndarray:
        """
        Read the next frame from the queue (blocking).

        Returns:
            The next frame as a numpy array.
        """
        return self.queue.get()

    def running(self) -> bool:
        """
        Check if capture is still running or has frames available.

        Returns:
            True if there are frames available or capture is not stopped.
        """
        return self.more() or not self.stopped

    def more(self) -> bool:
        """
        Check if there are frames available in the queue.

        Waits up to 0.5 seconds for frames to become available.

        Returns:
            True if frames are available in the queue.
        """
        tries = 0
        while self.queue.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1
        return self.queue.qsize() > 0

    def stop(self) -> None:
        """
        Stop the capture and release resources.

        Stops the capture thread and waits for it to finish.
        """
        self.stopped = True
        self.thread.join()

    def __enter__(self) -> FileVideoCapture:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


class StreamCapture:
    """
    Threaded video stream capture class for RTSP/RTMP/HTTP streams with
    automatic reconnection.

    This class is optimized for network video streams, providing robust handling
    of network interruptions, automatic reconnection with exponential backoff,
    and watchdog monitoring. It uses threading and queue buffering for optimal
    performance.

    Attributes:
        stream_url: URL of the video stream (RTSP, RTMP, HTTP, etc.)
        queue: Thread-safe queue containing captured frames
        stopped: Flag indicating if capture should stop

    Examples:
        ```python
        import supervision as sv
        import cv2

        # Basic RTSP stream usage
        stream_url = "rtsp://username:password@192.168.1.100:554/stream"
        capture = sv.StreamCapture(stream_url).start()
        while capture.running():
            frame = capture.read()
            cv2.imshow("Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.stop()
        cv2.destroyAllWindows()

        # Using context manager
        with sv.StreamCapture("rtsp://192.168.1.100:554/stream") as capture:
            for _ in range(100):
                if capture.more():
                    frame = capture.read()
                    # Process frame...

        # With hardware acceleration and custom buffer
        capture = sv.StreamCapture(
            stream_url="rtsp://camera.local/stream",
            use_hw_accel=True,
            queue_size=256,
            buffer_size=1
        ).start()
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
        name: str = "StreamCapture",
        reconnect_delay: int = 5,
        max_reconnect_delay: int = 60,
    ):
        """
        Initialize the StreamCapture.

        Args:
            stream_url: URL of the video stream (rtsp://, rtmp://, http://, etc.)
            transport: Transport protocol for RTSP ("tcp" or "udp", default: "tcp")
            use_hw_accel: Whether to use hardware acceleration (FFMPEG backend)
            buffer_size: OpenCV buffer size (lower = less latency, default: 1)
            queue_size: Maximum number of frames to buffer in queue
            transform: Optional function to transform each frame before queuing
            name: Name for the capture instance (used in warnings)
            reconnect_delay: Initial delay in seconds between reconnection attempts
            max_reconnect_delay: Maximum delay in seconds for exponential backoff
        """
        self.stream_url = stream_url
        self.transport = transport
        self.use_hw_accel = use_hw_accel
        self.buffer_size = buffer_size
        self.queue: Queue = Queue(maxsize=queue_size)
        self.stopped = False
        self.transform = transform
        self.name = name
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.reconnect_attempts = 0
        self.total_reconnect_attempts = 0
        self.last_frame_time = time.time()
        self.cap = None

        self.initialize_capture()

        self.thread = Thread(target=self.update, args=(), name=name)
        self.thread.daemon = True

        self.watchdog_thread = Thread(target=self.watchdog, name=f"{name}_watchdog")
        self.watchdog_thread.daemon = True

    def initialize_capture(self) -> None:
        """
        Initialize or reinitialize the stream capture.

        Configures OpenCV VideoCapture with optimized settings for network streams.
        """
        backend = cv2.CAP_FFMPEG if self.use_hw_accel else cv2.CAP_ANY

        # Build RTSP options for better performance
        if self.stream_url.startswith("rtsp://"):
            # Set transport protocol via environment variable for FFMPEG
            import os

            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self.transport}|max_delay;500000"
            )

        self.cap = cv2.VideoCapture(self.stream_url, backend)

        if not self.cap.isOpened():
            raise OSError(f"Cannot open stream: {self.stream_url}")

        # Optimize settings for streaming
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        # Set timeouts (in milliseconds)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)  # 30 seconds
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 30000)  # 30 seconds

        if self.use_hw_accel:
            self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            self.cap.set(cv2.CAP_PROP_HW_DEVICE, 0)

    def start(self) -> StreamCapture:
        """
        Start the capture and watchdog threads.

        Returns:
            Self for method chaining.
        """
        self.thread.start()
        self.watchdog_thread.start()
        return self

    def update(self) -> None:
        """
        Main loop running in separate thread to continuously capture frames.

        This method runs until stopped flag is set. It handles reconnection
        on failures and applies optional transforms to frames.
        """
        consecutive_successful_reads = 0
        consecutive_failures = 0

        while not self.stopped:
            try:
                if not self.cap or not self.cap.isOpened():
                    self.reconnect()
                    continue

                # Clear buffer to get latest frame (reduce latency)
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except Exception:
                        pass

                ret, frame = self.cap.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    warnings.warn(
                        f"Error reading frame from {self.name} "
                        f"(consecutive failures: {consecutive_failures})"
                    )

                    # Reconnect after 3 consecutive failures
                    if consecutive_failures >= 3:
                        self.reconnect()
                        consecutive_failures = 0
                    continue

                # Successful read
                consecutive_failures = 0
                self.last_frame_time = time.time()

                if self.transform:
                    frame = self.transform(frame)

                self.queue.put(frame)

                consecutive_successful_reads += 1
                if consecutive_successful_reads >= 100:
                    self.reconnect_attempts = 0
                    consecutive_successful_reads = 0

            except Exception as e:
                warnings.warn(f"Unexpected error in update loop for {self.name}: {e!s}")
                self.reconnect()
                time.sleep(1)

    def reconnect(self) -> None:
        """
        Attempt to reconnect to the stream with exponential backoff.

        Uses exponential backoff up to max_reconnect_delay seconds between attempts.
        """
        self.total_reconnect_attempts += 1
        self.reconnect_attempts += 1

        # Calculate wait time with exponential backoff
        wait_time = min(
            self.max_reconnect_delay,
            self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)),
        )

        warnings.warn(
            f"Attempting to reconnect {self.name}... "
            f"(Attempt: {self.reconnect_attempts}, "
            f"Total: {self.total_reconnect_attempts}, "
            f"Wait: {wait_time}s)"
        )

        if self.cap:
            self.cap.release()

        time.sleep(wait_time)

        try:
            self.initialize_capture()
            if self.cap.isOpened():
                warnings.warn(
                    f"Reconnection successful for {self.name} "
                    f"after {self.total_reconnect_attempts} total attempts"
                )
                self.reconnect_attempts = 0
            else:
                raise OSError("Failed to open stream")
        except Exception as e:
            warnings.warn(f"Reconnection failed for {self.name}: {e!s}")

    def watchdog(self) -> None:
        """
        Watchdog thread that monitors frame reception and triggers reconnection.

        Checks every 5 seconds if frames have been received in the last 30 seconds.
        """
        while not self.stopped:
            if time.time() - self.last_frame_time > 30:
                warnings.warn(
                    f"Watchdog detected no frames for {self.name} "
                    f"(last frame: {time.time() - self.last_frame_time:.1f}s ago), "
                    "restarting stream"
                )
                self.reconnect()
            time.sleep(5)

    def read(self) -> np.ndarray:
        """
        Read the next frame from the queue (blocking).

        Returns:
            The next frame as a numpy array.
        """
        return self.queue.get()

    def running(self) -> bool:
        """
        Check if capture is still running or has frames available.

        Returns:
            True if there are frames available or capture is not stopped.
        """
        return self.more() or not self.stopped

    def more(self) -> bool:
        """
        Check if there are frames available in the queue.

        Waits up to 0.5 seconds for frames to become available.

        Returns:
            True if frames are available in the queue.
        """
        tries = 0
        while self.queue.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1
        return self.queue.qsize() > 0

    def stop(self) -> None:
        """
        Stop the capture and release resources.

        Stops both capture and watchdog threads and releases the stream capture.
        """
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        if self.watchdog_thread.is_alive():
            self.watchdog_thread.join(timeout=5)
        if self.cap:
            self.cap.release()

    def get_health(self) -> dict[str, bool | float | int]:
        """
        Get health status information about the stream capture.

        Returns:
            Dictionary containing:
                - is_running: Whether capture is active
                - is_capture_open: Whether cv2.VideoCapture is open
                - last_frame_time: Timestamp of last successful frame
                - reconnect_attempts: Current consecutive reconnection attempts
                - total_reconnect_attempts: Total reconnection attempts since start
                - queue_size: Number of frames in queue
        """
        return {
            "is_running": not self.stopped,
            "is_capture_open": self.cap.isOpened() if self.cap else False,
            "last_frame_time": self.last_frame_time,
            "reconnect_attempts": self.reconnect_attempts,
            "total_reconnect_attempts": self.total_reconnect_attempts,
            "queue_size": self.queue.qsize(),
        }

    def get(self, prop_id: int) -> float:
        """
        Get a property value from the underlying VideoCapture.

        Args:
            prop_id: OpenCV property identifier (e.g., cv2.CAP_PROP_FRAME_WIDTH)

        Returns:
            The property value, or 0.0 if capture is not initialized.
        """
        if self.cap:
            return self.cap.get(prop_id)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        """
        Set a property value on the underlying VideoCapture.

        Args:
            prop_id: OpenCV property identifier
            value: Value to set

        Returns:
            True if successful, False if capture is not initialized.
        """
        if self.cap:
            return self.cap.set(prop_id, value)
        return False

    def is_alive(self) -> bool:
        """
        Check if the stream capture is alive and functioning.

        Returns:
            True if not stopped and capture is open.
        """
        return not self.stopped and (self.cap.isOpened() if self.cap else False)

    def __enter__(self) -> StreamCapture:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
