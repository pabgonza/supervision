"""
Video capture with simultaneous display and saving demo.

This example shows how to capture from webcam, video file, or stream
and optionally save to a video file while displaying in real-time.

Usage:
    # Webcam display only (default)
    python examples/pipeline/webcam_save_demo.py

    # Webcam with saving
    python examples/pipeline/webcam_save_demo.py --output recording.mp4

    # Video file with saving
    python examples/pipeline/webcam_save_demo.py --source file --input video.mp4 --output processed.mp4

    # Stream with saving
    python examples/pipeline/webcam_save_demo.py --source stream --input rtsp://camera/stream --output stream.mp4

    # Save without display
    python examples/pipeline/webcam_save_demo.py --output recording.mp4 --no-display

    # With FPS display
    python examples/pipeline/webcam_save_demo.py --output recording.mp4 --show-fps

    # Custom resolution (webcam only)
    python examples/pipeline/webcam_save_demo.py --width 1920 --height 1080 --output hd_recording.mp4

Press 'q' or ESC to stop.
"""

import argparse

import supervision as sv

import utils


def main():
    parser = argparse.ArgumentParser(
        description="Video capture with optional display and saving"
    )

    # Add standard arguments using utils
    utils.add_source_arguments(parser)
    utils.add_sink_arguments(parser)
    utils.add_fps_arguments(parser)

    # Add video codec argument (optional)
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="Video codec FOURCC (default: mp4v)",
    )

    args = parser.parse_args()

    # Print configuration
    print("Video Capture Demo")
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
    sink = utils.create_sink_from_args(args, "Video Capture", fps, width, height)
    pipeline = pipeline | sink

    # Run pipeline
    try:
        pipeline.run()
    except (KeyboardInterrupt, StopIteration):
        if args.output:
            print(f"\nCapture stopped. Video saved to: {args.output}")
        else:
            print("\nCapture stopped.")


if __name__ == "__main__":
    main()
