"""
Demo: Webcam capture with simultaneous display and video saving.

This example shows how to use MultiSink to both display frames in real-time
and save them to a video file simultaneously.

Usage:
    # Save with default settings (output.mp4, 640x480, 30fps)
    python examples/webcam_save_demo.py

    # Custom output file and resolution
    python examples/webcam_save_demo.py --output my_video.mp4 --width 1920 --height 1080

    # Specific camera
    python examples/webcam_save_demo.py --camera 1

    # With FPS display
    python examples/webcam_save_demo.py --show-fps
"""

import argparse

import supervision as sv


def main():
    parser = argparse.ArgumentParser(
        description="Capture from webcam, display and save to video file"
    )

    parser.add_argument("--camera", type=int, default=0, help="Camera ID (default: 0)")

    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Output video file path (default: output.mp4)",
    )

    parser.add_argument(
        "--width", type=int, default=640, help="Video width (default: 640)"
    )

    parser.add_argument(
        "--height", type=int, default=480, help="Video height (default: 480)"
    )

    parser.add_argument("--fps", type=int, default=30, help="FPS (default: 30)")

    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="Video codec FOURCC (default: mp4v)",
    )

    parser.add_argument("--show-fps", action="store_true", help="Display FPS on screen")

    args = parser.parse_args()

    print(f"Starting webcam capture (camera {args.camera})")
    print(f"Output: {args.output} ({args.width}x{args.height} @ {args.fps} fps)")
    print("Press 'q' or ESC to stop recording\n")

    # Create multi-sink with display and video file
    multi_sink = sv.MultiSink(
        [
            sv.DisplaySink("Webcam Recording", show_fps=args.show_fps),
            sv.VideoFileSink(
                output_path=args.output,
                fps=args.fps,
                width=args.width,
                height=args.height,
                codec=args.codec,
            ),
        ]
    )

    # Build pipeline
    pipeline = sv.Pipeline(
        sv.WebcamSource(camera_id=args.camera, width=args.width, height=args.height)
    )

    # Add FPS calculator if requested
    if args.show_fps:
        pipeline = pipeline | sv.FPSCalculatorStep()

    # Add multi-sink
    pipeline = pipeline | multi_sink

    # Run pipeline
    try:
        pipeline.run()
    except (KeyboardInterrupt, StopIteration):
        print(f"\nRecording stopped. Video saved to: {args.output}")


if __name__ == "__main__":
    main()
