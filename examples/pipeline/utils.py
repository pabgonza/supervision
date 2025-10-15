"""
Common utilities for pipeline examples.

This module provides reusable argument parsing and factory functions
to ensure consistent interfaces across all pipeline examples.
"""

import argparse
from typing import Optional, Union

import supervision as sv


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add standard source arguments to argument parser.

    Args:
        parser: ArgumentParser instance to add arguments to

    Adds:
        --source: Source type (webcam, file, stream)
        --input: Path to video file or stream URL
        --camera: Camera ID for webcam source
        --width: Camera width (webcam only)
        --height: Camera height (webcam only)
    """
    parser.add_argument(
        "--source",
        type=str,
        choices=["webcam", "file", "stream"],
        default="webcam",
        help="Source type (default: webcam)",
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Path to video file (for file source) or stream URL (for stream source)",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera ID (default: 0, for webcam source)",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Camera width (default: camera default, for webcam source)",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Camera height (default: camera default, for webcam source)",
    )


def add_sink_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add standard sink arguments to argument parser.

    Args:
        parser: ArgumentParser instance to add arguments to

    Adds:
        --output: Output video file path
        --no-display: Don't display video window
    """
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video file path (optional, e.g., output.mp4)",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display video window (useful with --output)",
    )


def add_fps_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add FPS display arguments to argument parser.

    Args:
        parser: ArgumentParser instance to add arguments to

    Adds:
        --show-fps: Display FPS counter
    """
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Display FPS counter",
    )


def create_source_from_args(args: argparse.Namespace) -> Union[sv.WebcamSource, sv.VideoFileSource, sv.StreamSource]:
    """
    Create supervision source from parsed arguments.

    Args:
        args: Namespace from argparse.parse_args() with source arguments

    Returns:
        Supervision source (WebcamSource, VideoFileSource, or StreamSource)

    Raises:
        ValueError: If input is required but not provided, or source type is invalid
    """
    if args.source == "webcam":
        webcam_kwargs = {"camera_id": args.camera}
        if hasattr(args, "width") and args.width is not None:
            webcam_kwargs["width"] = args.width
        if hasattr(args, "height") and args.height is not None:
            webcam_kwargs["height"] = args.height
        return sv.WebcamSource(**webcam_kwargs)

    elif args.source == "file":
        if not args.input:
            raise ValueError("--input required for file source")
        return sv.VideoFileSource(args.input)

    elif args.source == "stream":
        if not args.input:
            raise ValueError("--input required for stream source")
        return sv.StreamSource(args.input)

    else:
        raise ValueError(f"Unknown source type: {args.source}")


def create_sink_from_args(
    args: argparse.Namespace,
    window_name: str = "Pipeline",
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    codec: str = "mp4v",
) -> Union[sv.DisplaySink, sv.VideoFileSink, sv.MultiSink]:
    """
    Create supervision sink(s) from parsed arguments.

    Args:
        args: Namespace from argparse.parse_args() with sink arguments
        window_name: Name for display window
        fps: FPS for video file output (default: 30)
        width: Width for video file output (default: 1280)
        height: Height for video file output (default: 720)
        codec: Video codec FOURCC (default: mp4v)

    Returns:
        Supervision sink (DisplaySink, VideoFileSink, or MultiSink)

    Behavior:
        - No --output: DisplaySink only
        - With --output, no --no-display: MultiSink (DisplaySink + VideoFileSink)
        - With --output and --no-display: VideoFileSink only

    Note:
        For best results, pass the actual fps/width/height from your source.
        Defaults work for most 720p streams at 30fps.
    """
    show_fps = getattr(args, "show_fps", False)
    output = getattr(args, "output", None)
    no_display = getattr(args, "no_display", False)

    # Only save to file (no display)
    if output and no_display:
        return sv.VideoFileSink(
            output_path=output,
            fps=fps,
            width=width,
            height=height,
            codec=codec,
        )

    # Both display and save
    elif output:
        sinks = [sv.DisplaySink(window_name, show_fps=show_fps)]
        sinks.append(sv.VideoFileSink(
            output_path=output,
            fps=fps,
            width=width,
            height=height,
            codec=codec,
        ))
        return sv.MultiSink(sinks)

    # Only display
    else:
        return sv.DisplaySink(window_name, show_fps=show_fps)


def get_video_info_from_source(source: Union[sv.WebcamSource, sv.VideoFileSource, sv.StreamSource]) -> tuple:
    """
    Get video information (fps, width, height) from a source without consuming it.

    Args:
        source: Supervision source

    Returns:
        Tuple of (fps, width, height). Returns defaults (30, 1280, 720) if unable to read.
    """
    import cv2

    try:
        # Access the underlying VideoCapture object
        cap = source.cap if hasattr(source, 'cap') else None

        if cap is not None and cap.isOpened():
            # Get properties from VideoCapture
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Validate values
            if fps <= 0 or fps > 120:  # Invalid or unreasonable FPS
                fps = 30
            if width <= 0 or height <= 0:  # Invalid dimensions
                width, height = 1280, 720

            return int(fps), width, height
    except Exception as e:
        # If anything fails, return defaults
        pass

    # Fallback to reasonable defaults
    return 30, 1280, 720


def print_source_info(args: argparse.Namespace) -> None:
    """
    Print source configuration information.

    Args:
        args: Namespace from argparse.parse_args() with source arguments
    """
    print(f"Source: {args.source}")

    if args.source == "webcam":
        if hasattr(args, "width") and hasattr(args, "height") and args.width and args.height:
            print(f"Camera: {args.camera} ({args.width}x{args.height})")
        else:
            print(f"Camera: {args.camera} (default resolution)")

    elif args.source == "file":
        print(f"Input file: {args.input}")

    elif args.source == "stream":
        print(f"Stream URL: {args.input}")

    if hasattr(args, "output") and args.output:
        print(f"Output: {args.output}")
        if hasattr(args, "no_display") and args.no_display:
            print("Display: Disabled")
        else:
            print("Display: Enabled")
    else:
        print("Output: None (display only)")

    print("Press 'q' or ESC to quit\n")


def print_tracker_metrics(metrics: dict, title: str = "Tracking Metrics") -> None:
    """
    Print tracking performance metrics in a formatted table.

    Args:
        metrics: Dictionary from ByteTrackerStep.get_metrics()
        title: Title for the metrics section (default: "Tracking Metrics")
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Frames Processed:     {metrics['frames_processed']}")
    print(f"Avg Tracking Time:    {metrics['avg_processing_time_ms']:.2f} ms")
    print(f"Min Tracking Time:    {metrics['min_processing_time_ms']:.2f} ms")
    print(f"Max Tracking Time:    {metrics['max_processing_time_ms']:.2f} ms")
    print(f"Total Tracking Time:  {metrics['total_processing_time_s']:.2f} s")
    print("=" * 60)


def get_tracker_step(pipeline) -> Optional[sv.ByteTrackerStep]:
    """
    Get ByteTrackerStep from a pipeline.

    Args:
        pipeline: Supervision pipeline object

    Returns:
        ByteTrackerStep instance if found, None otherwise
    """
    if not hasattr(pipeline, "_steps"):
        return None

    for step in pipeline._steps:
        if isinstance(step, sv.ByteTrackerStep):
            return step

    return None
