"""
Simple pipeline example with video capture.

This demonstrates minimal code to capture from webcam, video file, or stream,
with optional display and video saving.

Usage:
    # Webcam with display only (default)
    python examples/pipeline/webcam_simple.py

    # Video file
    python examples/pipeline/webcam_simple.py --source file --input video.mp4

    # RTSP stream
    python examples/pipeline/webcam_simple.py --source stream --input rtsp://camera/stream

    # Save to video file
    python examples/pipeline/webcam_simple.py --output recording.mp4

    # Save without display
    python examples/pipeline/webcam_simple.py --output recording.mp4 --no-display

    # With FPS display
    python examples/pipeline/webcam_simple.py --show-fps

Press 'q' or ESC to stop.
"""

import argparse

import supervision as sv

import utils


def main():
    parser = argparse.ArgumentParser(
        description="Simple pipeline example with video capture"
    )

    # Add standard arguments using utils
    utils.add_source_arguments(parser)
    utils.add_sink_arguments(parser)
    utils.add_fps_arguments(parser)

    args = parser.parse_args()

    # Print configuration
    print("Simple Pipeline Demo")
    print("-" * 40)
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

    # Add sink(s) with video info from source
    sink = utils.create_sink_from_args(args, "Simple Pipeline", fps, width, height)
    pipeline = pipeline | sink

    # Run pipeline
    try:
        pipeline.run()
    except (KeyboardInterrupt, StopIteration):
        if args.output:
            print(f"\nStopped. Video saved to: {args.output}")
        else:
            print("\nStopped.")


if __name__ == "__main__":
    main()
