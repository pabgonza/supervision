"""
Pipeline demo showcasing the modern pipeline API in supervision.

This demo shows how to use the Pipeline system to build composable
video processing workflows.

Usage:
    # Webcam with FPS display
    python examples/pipeline_demo.py --source webcam

    # Video file with FPS
    python examples/pipeline_demo.py --source file --input video.mp4

    # RTSP stream
    python examples/pipeline_demo.py --source stream --input rtsp://192.168.1.100:554/stream

    # With resize
    python examples/pipeline_demo.py --source webcam --resize 1280 720
"""

import argparse
from typing import Optional

import supervision as sv


def webcam_demo(
    camera_id: int = 0, show_fps: bool = True, resize: Optional[tuple] = None
):
    """
    Demo pipeline with webcam source.

    Args:
        camera_id: Camera device ID
        show_fps: Whether to display FPS
        resize: Optional (width, height) tuple for resizing
    """
    print(f"Starting webcam pipeline (camera {camera_id})")
    print("Press 'q' or ESC to quit\n")

    # Build pipeline
    pipeline = sv.Pipeline(sv.WebcamSource(camera_id=camera_id))

    # Add FPS calculator
    if show_fps:
        pipeline = pipeline | sv.FPSCalculatorStep()

    # Add resize if requested
    if resize:
        width, height = resize
        pipeline = pipeline | sv.ResizeStep(width=width, height=height)

    # Add display sink
    pipeline = pipeline | sv.DisplaySink("Webcam Pipeline", show_fps=show_fps)

    # Run pipeline
    try:
        pipeline.run()
    except StopIteration:
        print("\nPipeline stopped by user")


def file_demo(video_path: str, show_fps: bool = True, resize: Optional[tuple] = None):
    """
    Demo pipeline with video file source.

    Args:
        video_path: Path to video file
        show_fps: Whether to display FPS
        resize: Optional (width, height) tuple for resizing
    """
    print(f"Starting file pipeline: {video_path}")
    print("Press 'q' or ESC to quit\n")

    # Build pipeline
    pipeline = sv.Pipeline(sv.VideoFileSource(video_path))

    # Add FPS calculator
    if show_fps:
        pipeline = pipeline | sv.FPSCalculatorStep()

    # Add resize if requested
    if resize:
        width, height = resize
        pipeline = pipeline | sv.ResizeStep(width=width, height=height)

    # Add display sink
    pipeline = pipeline | sv.DisplaySink("File Pipeline", show_fps=show_fps)

    # Run pipeline
    try:
        pipeline.run()
    except StopIteration:
        print("\nPipeline stopped")


def stream_demo(
    stream_url: str,
    transport: str = "tcp",
    show_fps: bool = True,
    resize: Optional[tuple] = None,
):
    """
    Demo pipeline with RTSP/RTMP/HTTP stream source.

    Args:
        stream_url: URL of the stream
        transport: Transport protocol (tcp/udp)
        show_fps: Whether to display FPS
        resize: Optional (width, height) tuple for resizing
    """
    print(f"Starting stream pipeline: {stream_url}")
    print(f"Transport: {transport}")
    print("Press 'q' or ESC to quit\n")

    # Build pipeline
    pipeline = sv.Pipeline(
        sv.StreamSource(stream_url=stream_url, transport=transport, buffer_size=1)
    )

    # Add FPS calculator
    if show_fps:
        pipeline = pipeline | sv.FPSCalculatorStep()

    # Add resize if requested
    if resize:
        width, height = resize
        pipeline = pipeline | sv.ResizeStep(width=width, height=height)

    # Add display sink
    pipeline = pipeline | sv.DisplaySink("Stream Pipeline", show_fps=show_fps)

    # Run pipeline
    try:
        pipeline.run()
    except StopIteration:
        print("\nPipeline stopped")


def callback_demo(camera_id: int = 0):
    """
    Demo pipeline with callback step for custom processing.

    Args:
        camera_id: Camera device ID
    """
    print(f"Starting callback demo (camera {camera_id})")
    print("This demo shows frame info in console\n")

    frame_count = 0

    def log_frame_info(data):
        nonlocal frame_count
        frame_count += 1
        if frame_count % 30 == 0:  # Log every 30 frames
            fps = data.get("fps", 0)
            frame_num = data.get("frame_number", 0)
            print(f"Frame {frame_num}: FPS={fps:.1f}")

    # Build pipeline with callback
    pipeline = (
        sv.Pipeline(sv.WebcamSource(camera_id=camera_id))
        | sv.FPSCalculatorStep()
        | sv.CallbackStep(log_frame_info)
        | sv.DisplaySink("Callback Demo", show_fps=True)
    )

    # Run pipeline
    try:
        pipeline.run()
    except StopIteration:
        print(f"\nTotal frames processed: {frame_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline demo for supervision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--source",
        type=str,
        choices=["webcam", "file", "stream", "callback"],
        default="webcam",
        help="Source type (default: webcam)",
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Input path (video file or stream URL)",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera ID for webcam source (default: 0)",
    )

    parser.add_argument(
        "--transport",
        type=str,
        choices=["tcp", "udp"],
        default="tcp",
        help="Transport for RTSP streams (default: tcp)",
    )

    parser.add_argument(
        "--no-fps",
        action="store_true",
        help="Disable FPS display",
    )

    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize frames to WIDTH HEIGHT",
    )

    args = parser.parse_args()

    show_fps = not args.no_fps
    resize = tuple(args.resize) if args.resize else None

    try:
        if args.source == "webcam":
            webcam_demo(args.camera, show_fps, resize)
        elif args.source == "file":
            if not args.input:
                print("Error: --input required for file source")
                return
            file_demo(args.input, show_fps, resize)
        elif args.source == "stream":
            if not args.input:
                print("Error: --input required for stream source")
                return
            stream_demo(args.input, args.transport, show_fps, resize)
        elif args.source == "callback":
            callback_demo(args.camera)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
