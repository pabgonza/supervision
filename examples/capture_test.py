"""
Test script for WebcamVideoCapture, FileVideoCapture, and StreamCapture.

This script demonstrates the usage of the threaded video capture classes
from supervision library.

Usage:
    # Test webcam (default camera 0)
    python examples/capture_test.py

    # Test specific camera
    python examples/capture_test.py --camera 1

    # Test video file
    python examples/capture_test.py --file path/to/video.mp4

    # Test RTSP stream
    python examples/capture_test.py --stream rtsp://username:password@192.168.1.100:554/stream

    # Test RTSP stream with TCP transport
    python examples/capture_test.py --stream rtsp://camera.local/stream --transport tcp
"""

import argparse
import time

import cv2

import supervision as sv


def test_webcam_capture(camera_id: int = 0, duration: int = 10):
    """
    Test WebcamVideoCapture with webcam or IP camera.

    Args:
        camera_id: Camera index or URL
        duration: How long to run the test (seconds)
    """
    print(f"Testing WebcamVideoCapture with camera {camera_id}")
    print(f"Will run for {duration} seconds")
    print("Press 'q' to quit early\n")

    # Create FPS monitor
    fps_monitor = sv.FPSMonitor(sample_size=30)

    # Initialize capture with context manager
    with sv.WebcamVideoCapture(
        src=camera_id, width=1280, height=720, name=f"Camera_{camera_id}"
    ) as capture:
        start_time = time.time()
        frame_count = 0

        while capture.running() and (time.time() - start_time) < duration:
            if capture.more():
                frame = capture.read()
                frame_count += 1
                fps_monitor.tick()

                # Get health status
                health = capture.get_health()

                # Draw info on frame
                info_text = [
                    f"Frame: {frame_count}",
                    f"FPS: {fps_monitor.fps:.1f}",
                    f"Queue: {health['queue_size']}",
                    f"Alive: {capture.is_alive()}",
                ]

                y_offset = 30
                for i, text in enumerate(info_text):
                    cv2.putText(
                        frame,
                        text,
                        (10, y_offset + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                # Display frame
                cv2.imshow("WebcamVideoCapture Test", frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nUser requested quit")
                    break
            else:
                time.sleep(0.01)

    cv2.destroyAllWindows()

    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\n{'=' * 50}")
    print("Test Summary:")
    print(f"{'=' * 50}")
    print(f"Total frames captured: {frame_count}")
    print(f"Elapsed time: {elapsed_time:.2f}s")
    print(f"Average FPS: {frame_count / elapsed_time:.2f}")
    print(f"Final health status: {health}")


def test_file_capture(video_path: str):
    """
    Test FileVideoCapture with video file.

    Args:
        video_path: Path to video file
    """
    print(f"Testing FileVideoCapture with file: {video_path}")
    print("Press 'q' to quit\n")

    # Get video info
    video_info = sv.VideoInfo.from_video_path(video_path)
    print(f"Video info: {video_info.width}x{video_info.height} @ {video_info.fps}fps")
    print(f"Total frames: {video_info.total_frames}\n")

    # Create FPS monitor
    fps_monitor = sv.FPSMonitor(sample_size=30)

    # Initialize capture with context manager
    with sv.FileVideoCapture(src=video_path, name="FileCapture") as capture:
        frame_count = 0
        start_time = time.time()

        while capture.running():
            if capture.more():
                frame = capture.read()
                frame_count += 1
                fps_monitor.tick()

                # Draw info on frame
                progress = (frame_count / video_info.total_frames) * 100
                info_text = [
                    f"Frame: {frame_count}/{video_info.total_frames}",
                    f"Progress: {progress:.1f}%",
                    f"Processing FPS: {fps_monitor.fps:.1f}",
                ]

                y_offset = 30
                for i, text in enumerate(info_text):
                    cv2.putText(
                        frame,
                        text,
                        (10, y_offset + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                # Display frame
                cv2.imshow("FileVideoCapture Test", frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nUser requested quit")
                    break
            else:
                time.sleep(0.01)

    cv2.destroyAllWindows()

    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\n{'=' * 50}")
    print("Test Summary:")
    print(f"{'=' * 50}")
    print(f"Total frames processed: {frame_count}/{video_info.total_frames}")
    print(f"Elapsed time: {elapsed_time:.2f}s")
    print(f"Processing FPS: {frame_count / elapsed_time:.2f}")
    print(f"Speedup vs real-time: {(frame_count / elapsed_time) / video_info.fps:.2f}x")


def test_stream_capture(stream_url: str, transport: str = "tcp", duration: int = 30):
    """
    Test StreamCapture with RTSP/RTMP/HTTP stream.

    Args:
        stream_url: URL of the stream
        transport: Transport protocol (tcp/udp)
        duration: How long to run the test (seconds)
    """
    print(f"Testing StreamCapture with URL: {stream_url}")
    print(f"Transport: {transport}")
    print(f"Will run for {duration} seconds")
    print("Press 'q' to quit early\n")

    # Create FPS monitor
    fps_monitor = sv.FPSMonitor(sample_size=30)

    # Initialize capture with context manager
    with sv.StreamCapture(
        stream_url=stream_url, transport=transport, buffer_size=1, name="StreamTest"
    ) as capture:
        start_time = time.time()
        frame_count = 0

        while capture.running() and (time.time() - start_time) < duration:
            if capture.more():
                frame = capture.read()
                frame_count += 1
                fps_monitor.tick()

                # Get health status
                health = capture.get_health()

                # Draw info on frame
                info_text = [
                    f"Frame: {frame_count}",
                    f"FPS: {fps_monitor.fps:.1f}",
                    f"Queue: {health['queue_size']}",
                    f"Reconnects: {health['total_reconnect_attempts']}",
                    f"Alive: {capture.is_alive()}",
                ]

                y_offset = 30
                for i, text in enumerate(info_text):
                    cv2.putText(
                        frame,
                        text,
                        (10, y_offset + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                # Display frame
                cv2.imshow("StreamCapture Test", frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nUser requested quit")
                    break
            else:
                time.sleep(0.01)

    cv2.destroyAllWindows()

    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\n{'=' * 50}")
    print("Test Summary:")
    print(f"{'=' * 50}")
    print(f"Total frames captured: {frame_count}")
    print(f"Elapsed time: {elapsed_time:.2f}s")
    print(f"Average FPS: {frame_count / elapsed_time:.2f}")
    print(f"Final health status: {health}")


def main():
    parser = argparse.ArgumentParser(
        description="Test WebcamVideoCapture, FileVideoCapture, and StreamCapture"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index or ID (default: 0)"
    )
    parser.add_argument("--file", type=str, help="Path to video file to test")
    parser.add_argument(
        "--stream", type=str, help="Stream URL (rtsp://, rtmp://, http://)"
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="tcp",
        choices=["tcp", "udp"],
        help="Transport protocol for RTSP (default: tcp)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Test duration in seconds (default: 10)",
    )

    args = parser.parse_args()

    try:
        if args.stream:
            test_stream_capture(args.stream, args.transport, args.duration)
        elif args.file:
            test_file_capture(args.file)
        else:
            test_webcam_capture(args.camera, args.duration)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during test: {e}")
        raise


if __name__ == "__main__":
    main()
