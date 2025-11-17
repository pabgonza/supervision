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


class OakRgbCapture:
    """
    Threaded capture class for Luxonis OAK RGB cameras using DepthAI API v3.

    This class captures RGB frames from OAK cameras (OAK-1, or the RGB sensor
    on OAK-D/OAK-D Lite/OAK-D Pro) in a separate thread using a queue buffer.
    Supports auto-detection of devices and manual camera controls.

    Attributes:
        device_mxid: Device MxId (None for auto-detection)
        queue: Thread-safe queue containing captured frames
        stopped: Flag indicating if capture should stop
        device_info: Information about the connected device

    Examples:
        ```python
        import supervision as sv
        import cv2

        # Basic usage with auto-detection
        capture = sv.OakRgbCapture().start()
        while capture.running():
            frame = capture.read()
            cv2.imshow("OAK RGB", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.stop()
        cv2.destroyAllWindows()

        # With custom resolution and FPS
        capture = sv.OakRgbCapture(
            width=1920,
            height=1080,
            fps=30
        ).start()

        # With specific device
        capture = sv.OakRgbCapture(
            device_mxid="14442C10D13EAFD000"
        ).start()

        # With manual controls
        capture = sv.OakRgbCapture(
            manual_exposure=10000,  # 10ms
            manual_iso=400
        ).start()
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
        name: str = "OakRgbCapture",
    ):
        """
        Initialize the OakRgbCapture.

        Args:
            device_mxid: Device MxId for specific device (None for auto-detection)
            width: Frame width in pixels (max 3840 for 4K)
            height: Frame height in pixels (max 2160 for 4K)
            fps: Frames per second (e.g., 30, 60)
            color_order: Color order "RGB" or "BGR" (default: "BGR" for OpenCV)
            interleaved: True for RGBRGBRGB..., False for RRR...GGG...BBB...
            manual_exposure: Manual exposure time in microseconds (1-33000,
                None for auto)
            manual_iso: Manual ISO sensitivity (100-1600, None for auto)
            manual_focus: Manual focus value (0-255, None for auto)
            manual_white_balance: Manual white balance in Kelvin (1000-12000,
                None for auto)
            transform: Optional function to transform each frame before queuing
            queue_size: Maximum number of frames to buffer in queue
            name: Name for the capture instance
        """
        try:
            import depthai as dai
        except ImportError:
            raise ImportError(
                "depthai is required for OAK camera support. "
                "Install it with: pip install supervision[oak]"
            )

        self.dai = dai
        self.device_mxid = device_mxid
        self.width = width
        self.height = height
        self.fps = fps
        self.color_order = color_order
        self.interleaved = interleaved
        self.manual_exposure = manual_exposure
        self.manual_iso = manual_iso
        self.manual_focus = manual_focus
        self.manual_white_balance = manual_white_balance
        self.transform = transform
        self.name = name
        self.queue: Queue = Queue(maxsize=queue_size)
        self.stopped = False
        self.device = None
        self.device_info = None

        self._initialize_device()

        self.thread = Thread(target=self.update, args=(), name=name)
        self.thread.daemon = True

    def _initialize_device(self) -> None:
        """Initialize DepthAI device and pipeline."""
        devices = self.dai.Device.getAllAvailableDevices()
        if not devices:
            raise OSError("No OAK devices found")

        if self.device_mxid:
            device_info = None
            for dev in devices:
                dev_id = dev.getMxId() if hasattr(dev, "getMxId") else dev.deviceId
                if dev_id == self.device_mxid:
                    device_info = dev
                    break
            if not device_info:
                raise OSError(f"Device with MxId {self.device_mxid} not found")
        else:
            device_info = devices[0]

        dev_id = (
            device_info.getMxId()
            if hasattr(device_info, "getMxId")
            else device_info.deviceId
        )
        self.device_info = {
            "mxid": dev_id,
            "state": str(device_info.state),
            "protocol": str(device_info.protocol),
        }

        pipeline = self.dai.Pipeline()

        cam = pipeline.create(self.dai.node.ColorCamera)
        cam.setPreviewSize(self.width, self.height)
        cam.setFps(self.fps)

        if self.color_order == "BGR":
            cam.setColorOrder(self.dai.ColorCameraProperties.ColorOrder.BGR)
        else:
            cam.setColorOrder(self.dai.ColorCameraProperties.ColorOrder.RGB)

        cam.setInterleaved(self.interleaved)

        xout = pipeline.create(self.dai.node.XLinkOut)
        xout.setStreamName("rgb")
        cam.preview.link(xout.input)

        if any(
            [
                self.manual_exposure is not None,
                self.manual_iso is not None,
                self.manual_focus is not None,
                self.manual_white_balance is not None,
            ]
        ):
            control_in = pipeline.create(self.dai.node.XLinkIn)
            control_in.setStreamName("control")
            control_in.out.link(cam.inputControl)

        self.device = self.dai.Device(pipeline, device_info)
        self.output_queue = self.device.getOutputQueue("rgb", maxSize=4, blocking=False)

        if any(
            [
                self.manual_exposure is not None,
                self.manual_iso is not None,
                self.manual_focus is not None,
                self.manual_white_balance is not None,
            ]
        ):
            ctrl = self.dai.CameraControl()
            if self.manual_exposure is not None and self.manual_iso is not None:
                ctrl.setManualExposure(self.manual_exposure, self.manual_iso)
            if self.manual_focus is not None:
                ctrl.setManualFocus(self.manual_focus)
            if self.manual_white_balance is not None:
                ctrl.setManualWhiteBalance(self.manual_white_balance)

            control_queue = self.device.getInputQueue("control")
            control_queue.send(ctrl)

    def start(self) -> OakRgbCapture:
        """
        Start the capture thread.

        Returns:
            Self for method chaining.
        """
        self.thread.start()
        return self

    def update(self) -> None:
        """
        Main loop running in separate thread to continuously capture frames.

        This method runs until stopped flag is set.
        """
        while not self.stopped:
            try:
                img_frame = self.output_queue.get()
                if img_frame is None:
                    continue

                frame = img_frame.getCvFrame()

                if self.transform:
                    frame = self.transform(frame)

                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except Exception:
                        pass

                self.queue.put(frame)

            except Exception as e:
                warnings.warn(f"Error in {self.name} update loop: {e!s}")
                time.sleep(0.1)

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

        Stops the capture thread and closes the device connection.
        """
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        if self.device:
            self.device.close()

    def get_health(self) -> dict[str, bool | str | int]:
        """
        Get health status information about the capture.

        Returns:
            Dictionary containing:
                - is_running: Whether capture is active
                - is_device_connected: Whether device is connected
                - device_mxid: Device MxId
                - queue_size: Number of frames in queue
        """
        return {
            "is_running": not self.stopped,
            "is_device_connected": self.device is not None,
            "device_mxid": self.device_info.get("mxid") if self.device_info else None,
            "queue_size": self.queue.qsize(),
        }

    def get_device_info(self) -> dict[str, str]:
        """
        Get device information.

        Returns:
            Dictionary with device mxid, state, and protocol.
        """
        return self.device_info if self.device_info else {}

    def is_alive(self) -> bool:
        """
        Check if the capture is alive and functioning.

        Returns:
            True if not stopped and device is connected.
        """
        return not self.stopped and self.device is not None

    def __enter__(self) -> OakRgbCapture:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


class OakStereoCapture:
    """
    Threaded capture class for Luxonis OAK stereo cameras using DepthAI API v3.

    This class captures multiple streams simultaneously from OAK stereo cameras
    (OAK-D, OAK-D Lite, OAK-D Pro) including left/right mono cameras, depth,
    disparity, and rectified views.

    Attributes:
        device_mxid: Device MxId (None for auto-detection)
        output_streams: List of streams to capture
        queues: Dict of thread-safe queues for each stream
        stopped: Flag indicating if capture should stop
        device_info: Information about the connected device

    Examples:
        ```python
        import supervision as sv
        import cv2

        # Capture depth and disparity
        capture = sv.OakStereoCapture(
            output_streams=["depth", "disparity"]
        ).start()

        while capture.running():
            frames = capture.read()  # Dict with "depth" and "disparity"
            cv2.imshow("Depth", frames["depth"])
            cv2.imshow("Disparity", frames["disparity"])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.stop()

        # All streams with custom resolution
        capture = sv.OakStereoCapture(
            resolution="800p",
            output_streams=["left", "right", "depth", "rectified_left",
                            "rectified_right"],
            extended_disparity=True,
            subpixel=True
        ).start()
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
        name: str = "OakStereoCapture",
    ):
        """
        Initialize the OakStereoCapture.

        Args:
            device_mxid: Device MxId for specific device (None for auto-detection)
            resolution: Mono camera resolution ("400p", "480p", "720p", "800p")
            output_streams: List of streams to capture. Options:
                - "left": Left mono camera
                - "right": Right mono camera
                - "depth": Depth map
                - "disparity": Disparity map
                - "rectified_left": Rectified left mono
                - "rectified_right": Rectified right mono
                Default: ["depth"]
            extended_disparity: Enable extended disparity (closer depth range)
            subpixel: Enable subpixel mode (better precision at distance)
            lr_check: Enable left-right check (better occlusion handling)
            median_filter: Median filter kernel size (KERNEL_3x3, KERNEL_5x5,
                KERNEL_7x7)
            transform: Optional function to transform frames dict before queuing
            queue_size: Maximum number of frame dicts to buffer in queue
            name: Name for the capture instance
        """
        try:
            import depthai as dai
        except ImportError:
            raise ImportError(
                "depthai is required for OAK camera support. "
                "Install it with: pip install supervision[oak]"
            )

        self.dai = dai
        self.device_mxid = device_mxid
        self.resolution = resolution
        self.output_streams = output_streams or ["depth"]
        self.extended_disparity = extended_disparity
        self.subpixel = subpixel
        self.lr_check = lr_check
        self.median_filter = median_filter
        self.transform = transform
        self.name = name
        self.queue: Queue = Queue(maxsize=queue_size)
        self.stopped = False
        self.device = None
        self.device_info = None
        self.output_queues = {}

        valid_streams = [
            "left",
            "right",
            "depth",
            "disparity",
            "rectified_left",
            "rectified_right",
        ]
        for stream in self.output_streams:
            if stream not in valid_streams:
                raise ValueError(
                    f"Invalid output stream: {stream}. Valid options: {valid_streams}"
                )

        self._initialize_device()

        self.thread = Thread(target=self.update, args=(), name=name)
        self.thread.daemon = True

    def _initialize_device(self) -> None:
        """Initialize DepthAI device and pipeline."""
        devices = self.dai.Device.getAllAvailableDevices()
        if not devices:
            raise OSError("No OAK devices found")

        if self.device_mxid:
            device_info = None
            for dev in devices:
                dev_id = dev.getMxId() if hasattr(dev, "getMxId") else dev.deviceId
                if dev_id == self.device_mxid:
                    device_info = dev
                    break
            if not device_info:
                raise OSError(f"Device with MxId {self.device_mxid} not found")
        else:
            device_info = devices[0]

        dev_id = (
            device_info.getMxId()
            if hasattr(device_info, "getMxId")
            else device_info.deviceId
        )
        self.device_info = {
            "mxid": dev_id,
            "state": str(device_info.state),
            "protocol": str(device_info.protocol),
        }

        pipeline = self.dai.Pipeline()

        mono_left = pipeline.create(self.dai.node.MonoCamera)
        mono_left.setCamera("left")
        mono_right = pipeline.create(self.dai.node.MonoCamera)
        mono_right.setCamera("right")

        resolution_map = {
            "400p": self.dai.MonoCameraProperties.SensorResolution.THE_400_P,
            "480p": self.dai.MonoCameraProperties.SensorResolution.THE_480_P,
            "720p": self.dai.MonoCameraProperties.SensorResolution.THE_720_P,
            "800p": self.dai.MonoCameraProperties.SensorResolution.THE_800_P,
        }
        if self.resolution not in resolution_map:
            raise ValueError(
                f"Invalid resolution: {self.resolution}. "
                f"Valid options: {list(resolution_map.keys())}"
            )

        mono_left.setResolution(resolution_map[self.resolution])
        mono_right.setResolution(resolution_map[self.resolution])

        stereo = pipeline.create(self.dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(
            self.dai.node.StereoDepth.PresetMode.HIGH_DENSITY
        )

        stereo.setExtendedDisparity(self.extended_disparity)
        stereo.setSubpixel(self.subpixel)
        stereo.setLeftRightCheck(self.lr_check)

        median_map = {
            "KERNEL_3x3": self.dai.MedianFilter.KERNEL_3x3,
            "KERNEL_5x5": self.dai.MedianFilter.KERNEL_5x5,
            "KERNEL_7x7": self.dai.MedianFilter.KERNEL_7x7,
        }
        if self.median_filter in median_map:
            stereo.initialConfig.setMedianFilter(median_map[self.median_filter])

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        for stream in self.output_streams:
            xout = pipeline.create(self.dai.node.XLinkOut)
            xout.setStreamName(stream)

            if stream == "left":
                mono_left.out.link(xout.input)
            elif stream == "right":
                mono_right.out.link(xout.input)
            elif stream == "depth":
                stereo.depth.link(xout.input)
            elif stream == "disparity":
                stereo.disparity.link(xout.input)
            elif stream == "rectified_left":
                stereo.rectifiedLeft.link(xout.input)
            elif stream == "rectified_right":
                stereo.rectifiedRight.link(xout.input)

        self.device = self.dai.Device(pipeline, device_info)

        for stream in self.output_streams:
            self.output_queues[stream] = self.device.getOutputQueue(
                stream, maxSize=4, blocking=False
            )

    def start(self) -> OakStereoCapture:
        """
        Start the capture thread.

        Returns:
            Self for method chaining.
        """
        self.thread.start()
        return self

    def update(self) -> None:
        """
        Main loop running in separate thread to continuously capture frames.

        This method runs until stopped flag is set.
        """
        while not self.stopped:
            try:
                frames = {}
                all_available = True

                for stream, queue in self.output_queues.items():
                    img_frame = queue.tryGet()
                    if img_frame is None:
                        all_available = False
                        break

                    frame = img_frame.getCvFrame()
                    frames[stream] = frame

                if not all_available:
                    time.sleep(0.001)
                    continue

                if self.transform:
                    frames = self.transform(frames)

                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except Exception:
                        pass

                self.queue.put(frames)

            except Exception as e:
                warnings.warn(f"Error in {self.name} update loop: {e!s}")
                time.sleep(0.1)

    def read(self) -> dict[str, np.ndarray]:
        """
        Read the next frame dict from the queue (blocking).

        Returns:
            Dictionary mapping stream names to frames as numpy arrays.
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

        Stops the capture thread and closes the device connection.
        """
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        if self.device:
            self.device.close()

    def get_health(self) -> dict[str, bool | str | int]:
        """
        Get health status information about the capture.

        Returns:
            Dictionary containing health status.
        """
        return {
            "is_running": not self.stopped,
            "is_device_connected": self.device is not None,
            "device_mxid": self.device_info.get("mxid") if self.device_info else None,
            "queue_size": self.queue.qsize(),
            "active_streams": self.output_streams,
        }

    def get_device_info(self) -> dict[str, str]:
        """
        Get device information.

        Returns:
            Dictionary with device mxid, state, and protocol.
        """
        return self.device_info if self.device_info else {}

    def is_alive(self) -> bool:
        """
        Check if the capture is alive and functioning.

        Returns:
            True if not stopped and device is connected.
        """
        return not self.stopped and self.device is not None

    def __enter__(self) -> OakStereoCapture:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


class OakRgbDepthCapture:
    """
    Threaded capture class for Luxonis OAK cameras with aligned RGB + Depth.

    This class captures aligned RGB and depth streams from OAK-D cameras,
    providing the common use case of color image with corresponding depth map.

    Attributes:
        device_mxid: Device MxId (None for auto-detection)
        queue: Thread-safe queue containing RGB and depth frames
        stopped: Flag indicating if capture should stop
        device_info: Information about the connected device

    Examples:
        ```python
        import supervision as sv
        import cv2

        # Basic usage
        capture = sv.OakRgbDepthCapture().start()

        while capture.running():
            rgb, depth = capture.read()
            cv2.imshow("RGB", rgb)
            cv2.imshow("Depth", depth)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.stop()

        # With custom settings
        capture = sv.OakRgbDepthCapture(
            rgb_width=1920,
            rgb_height=1080,
            depth_resolution="800p",
            fps=30
        ).start()
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
        name: str = "OakRgbDepthCapture",
    ):
        """
        Initialize the OakRgbDepthCapture.

        Args:
            device_mxid: Device MxId for specific device (None for auto-detection)
            rgb_width: RGB frame width (max 3840)
            rgb_height: RGB frame height (max 2160)
            depth_resolution: Depth resolution ("400p", "480p", "720p", "800p")
            fps: Frames per second
            color_order: Color order "RGB" or "BGR"
            extended_disparity: Enable extended disparity for closer depth
            subpixel: Enable subpixel mode for better precision
            lr_check: Enable left-right check
            align_to_rgb: Align depth to RGB perspective (True recommended)
            manual_exposure: Manual exposure in microseconds (None for auto)
            manual_iso: Manual ISO (None for auto)
            manual_focus: Manual focus (None for auto)
            manual_white_balance: Manual white balance in Kelvin (None for auto)
            transform: Optional function to transform (rgb, depth) tuple
            queue_size: Maximum number of frame pairs to buffer
            name: Name for the capture instance
        """
        try:
            import depthai as dai
        except ImportError:
            raise ImportError(
                "depthai is required for OAK camera support. "
                "Install it with: pip install supervision[oak]"
            )

        self.dai = dai
        self.device_mxid = device_mxid
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height
        self.depth_resolution = depth_resolution
        self.fps = fps
        self.color_order = color_order
        self.extended_disparity = extended_disparity
        self.subpixel = subpixel
        self.lr_check = lr_check
        self.align_to_rgb = align_to_rgb
        self.manual_exposure = manual_exposure
        self.manual_iso = manual_iso
        self.manual_focus = manual_focus
        self.manual_white_balance = manual_white_balance
        self.transform = transform
        self.name = name
        self.queue: Queue = Queue(maxsize=queue_size)
        self.stopped = False
        self.device = None
        self.device_info = None

        self._initialize_device()

        self.thread = Thread(target=self.update, args=(), name=name)
        self.thread.daemon = True

    def _initialize_device(self) -> None:
        """Initialize DepthAI device and pipeline."""
        devices = self.dai.Device.getAllAvailableDevices()
        if not devices:
            raise OSError("No OAK devices found")

        if self.device_mxid:
            device_info = None
            for dev in devices:
                dev_id = dev.getMxId() if hasattr(dev, "getMxId") else dev.deviceId
                if dev_id == self.device_mxid:
                    device_info = dev
                    break
            if not device_info:
                raise OSError(f"Device with MxId {self.device_mxid} not found")
        else:
            device_info = devices[0]

        dev_id = (
            device_info.getMxId()
            if hasattr(device_info, "getMxId")
            else device_info.deviceId
        )
        self.device_info = {
            "mxid": dev_id,
            "state": str(device_info.state),
            "protocol": str(device_info.protocol),
        }

        pipeline = self.dai.Pipeline()

        cam_rgb = pipeline.create(self.dai.node.ColorCamera)
        cam_rgb.setPreviewSize(self.rgb_width, self.rgb_height)
        cam_rgb.setFps(self.fps)

        if self.color_order == "BGR":
            cam_rgb.setColorOrder(self.dai.ColorCameraProperties.ColorOrder.BGR)
        else:
            cam_rgb.setColorOrder(self.dai.ColorCameraProperties.ColorOrder.RGB)

        mono_left = pipeline.create(self.dai.node.MonoCamera)
        mono_left.setCamera("left")
        mono_right = pipeline.create(self.dai.node.MonoCamera)
        mono_right.setCamera("right")

        resolution_map = {
            "400p": self.dai.MonoCameraProperties.SensorResolution.THE_400_P,
            "480p": self.dai.MonoCameraProperties.SensorResolution.THE_480_P,
            "720p": self.dai.MonoCameraProperties.SensorResolution.THE_720_P,
            "800p": self.dai.MonoCameraProperties.SensorResolution.THE_800_P,
        }
        mono_left.setResolution(
            resolution_map.get(self.depth_resolution, resolution_map["400p"])
        )
        mono_right.setResolution(
            resolution_map.get(self.depth_resolution, resolution_map["400p"])
        )

        stereo = pipeline.create(self.dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(
            self.dai.node.StereoDepth.PresetMode.HIGH_DENSITY
        )
        stereo.setExtendedDisparity(self.extended_disparity)
        stereo.setSubpixel(self.subpixel)
        stereo.setLeftRightCheck(self.lr_check)

        if self.align_to_rgb:
            stereo.setDepthAlign(self.dai.CameraBoardSocket.CAM_A)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        xout_rgb = pipeline.create(self.dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        xout_depth = pipeline.create(self.dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        if any(
            [
                self.manual_exposure is not None,
                self.manual_iso is not None,
                self.manual_focus is not None,
                self.manual_white_balance is not None,
            ]
        ):
            control_in = pipeline.create(self.dai.node.XLinkIn)
            control_in.setStreamName("control")
            control_in.out.link(cam_rgb.inputControl)

        self.device = self.dai.Device(pipeline, device_info)
        self.rgb_queue = self.device.getOutputQueue("rgb", maxSize=4, blocking=False)
        self.depth_queue = self.device.getOutputQueue(
            "depth", maxSize=4, blocking=False
        )

        if any(
            [
                self.manual_exposure is not None,
                self.manual_iso is not None,
                self.manual_focus is not None,
                self.manual_white_balance is not None,
            ]
        ):
            ctrl = self.dai.CameraControl()
            if self.manual_exposure is not None and self.manual_iso is not None:
                ctrl.setManualExposure(self.manual_exposure, self.manual_iso)
            if self.manual_focus is not None:
                ctrl.setManualFocus(self.manual_focus)
            if self.manual_white_balance is not None:
                ctrl.setManualWhiteBalance(self.manual_white_balance)

            control_queue = self.device.getInputQueue("control")
            control_queue.send(ctrl)

    def start(self) -> OakRgbDepthCapture:
        """Start the capture thread."""
        self.thread.start()
        return self

    def update(self) -> None:
        """Main loop to continuously capture RGB and depth frames."""
        while not self.stopped:
            try:
                rgb_frame = self.rgb_queue.tryGet()
                depth_frame = self.depth_queue.tryGet()

                if rgb_frame is None or depth_frame is None:
                    time.sleep(0.001)
                    continue

                rgb = rgb_frame.getCvFrame()
                depth = depth_frame.getCvFrame()

                if self.transform:
                    rgb, depth = self.transform((rgb, depth))

                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except Exception:
                        pass

                self.queue.put((rgb, depth))

            except Exception as e:
                warnings.warn(f"Error in {self.name} update loop: {e!s}")
                time.sleep(0.1)

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Read the next RGB and depth frame pair (blocking).

        Returns:
            Tuple of (rgb_frame, depth_frame) as numpy arrays.
        """
        return self.queue.get()

    def running(self) -> bool:
        """Check if capture is still running."""
        return self.more() or not self.stopped

    def more(self) -> bool:
        """Check if frames are available."""
        tries = 0
        while self.queue.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1
        return self.queue.qsize() > 0

    def stop(self) -> None:
        """Stop capture and release resources."""
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        if self.device:
            self.device.close()

    def get_health(self) -> dict[str, bool | str | int]:
        """Get health status."""
        return {
            "is_running": not self.stopped,
            "is_device_connected": self.device is not None,
            "device_mxid": self.device_info.get("mxid") if self.device_info else None,
            "queue_size": self.queue.qsize(),
        }

    def get_device_info(self) -> dict[str, str]:
        """Get device information."""
        return self.device_info if self.device_info else {}

    def is_alive(self) -> bool:
        """Check if capture is alive."""
        return not self.stopped and self.device is not None

    def __enter__(self) -> OakRgbDepthCapture:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
