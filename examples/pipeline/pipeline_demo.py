"""
Pipeline demo showcasing the modern pipeline API in supervision.

This demo shows how to use the Pipeline system to build composable
video processing workflows with different sources and sinks.

Usage:
    # Webcam with FPS display
    python examples/pipeline/pipeline_demo.py --source webcam --show-fps

    # Video file
    python examples/pipeline/pipeline_demo.py --source file --input video.mp4

    # RTSP stream
    python examples/pipeline/pipeline_demo.py --source stream --input rtsp://192.168.1.100:554/stream

    # With output
    python examples/pipeline/pipeline_demo.py --source webcam --output recording.mp4

    # With resize
    python examples/pipeline/pipeline_demo.py --source webcam --resize 1280 720

    # Callback demo
    python examples/pipeline/pipeline_demo.py --callback

Press 'q' or ESC to quit.
"""

import argparse

import supervision as sv

import utils


def run_pipeline(
    args: argparse.Namespace,
    resize: tuple | None = None,
):
    """
    Run basic pipeline with source, optional resize, and sink.

    Args:
        args: Parsed arguments
        resize: Optional (width, height) tuple for resizing
    """
    print("Starting pipeline...")
    utils.print_source_info(args)

    # Create source
    source = utils.create_source_from_args(args)

    # Get video info from source for proper output configuration
    fps, width, height = utils.get_video_info_from_source(source)

    # Build pipeline
    pipeline = sv.Pipeline(source)

    # Add FPS calculator if requested
    if args.show_fps:
        pipeline = pipeline | sv.FPSCalculatorStep()

    # Add resize if requested
    if resize:
        width, height = resize
        pipeline = pipeline | sv.ResizeStep(width=width, height=height)

    # Add sink(s) with video info from source
    sink = utils.create_sink_from_args(args, "Pipeline Demo", fps, width, height)
    pipeline = pipeline | sink

    # Run pipeline
    try:
        pipeline.run()
    except (KeyboardInterrupt, StopIteration):
        if args.output:
            print(f"\nPipeline stopped. Video saved to: {args.output}")
        else:
            print("\nPipeline stopped.")


def run_callback_demo(args: argparse.Namespace):
    """
    Run pipeline with callback step for custom processing.

    Args:
        args: Parsed arguments
    """
    print("Starting callback demo...")
    print("This demo shows frame info in console\n")
    utils.print_source_info(args)

    frame_count = 0

    def log_frame_info(data):
        nonlocal frame_count
        frame_count += 1
        if frame_count % 30 == 0:  # Log every 30 frames
            fps = data.get("fps", 0)
            frame_num = data.get("frame_number", 0)
            print(f"Frame {frame_num}: FPS={fps:.1f}")

    # Create source
    source = utils.create_source_from_args(args)

    # Get video info from source for proper output configuration
    fps, width, height = utils.get_video_info_from_source(source)

    # Build pipeline with callback
    pipeline = (
        sv.Pipeline(source)
        | sv.FPSCalculatorStep()
        | sv.CallbackStep(log_frame_info)
    )

    # Add sink(s) with video info from source
    sink = utils.create_sink_from_args(args, "Callback Demo", fps, width, height)
    pipeline = pipeline | sink

    # Run pipeline
    try:
        pipeline.run()
    except (KeyboardInterrupt, StopIteration):
        print(f"\nTotal frames processed: {frame_count}")
        if args.output:
            print(f"Video saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline demo for supervision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add standard arguments
    utils.add_source_arguments(parser)
    utils.add_sink_arguments(parser)
    utils.add_fps_arguments(parser)

    # Add demo-specific arguments
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize frames to WIDTH HEIGHT",
    )

    parser.add_argument(
        "--callback",
        action="store_true",
        help="Run callback demo instead of basic pipeline",
    )

    parser.add_argument(
        "--transport",
        type=str,
        choices=["tcp", "udp"],
        default="tcp",
        help="Transport for RTSP streams (default: tcp)",
    )

    args = parser.parse_args()

    try:
        if args.callback:
            run_callback_demo(args)
        else:
            resize = tuple(args.resize) if args.resize else None
            run_pipeline(args, resize=resize)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
