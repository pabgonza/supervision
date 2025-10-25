"""
Common utilities for pipeline examples.

This module provides reusable argument parsing and factory functions
to ensure consistent interfaces across all pipeline examples.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Union

import yaml

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


def create_source(
    source_str: str,
    camera_id: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> Union[sv.WebcamSource, sv.VideoFileSource, sv.StreamSource]:
    """
    Create video source automatically detecting the type.

    Detects source type based on string pattern:
    - RTSP/HTTP streams: starts with 'rtsp://' or 'http://'
    - Video files: existing file path
    - Webcam ID: numeric string ('0', '1', etc.) or integer
    - Stream: fallback for other cases

    Args:
        source_str: Source string (URL, file path, or camera ID). Can be int for webcam.
        camera_id: Camera ID for webcam (used only if source_str is numeric)
        width: Optional width for webcam
        height: Optional height for webcam

    Returns:
        Appropriate supervision source

    Examples:
        >>> create_source('rtsp://192.168.1.100:554/stream')
        StreamSource(...)
        >>> create_source('video.mp4')
        VideoFileSource('video.mp4')
        >>> create_source('0')
        WebcamSource(camera_id=0)
        >>> create_source(0)
        WebcamSource(camera_id=0)
    """
    # Convert int to string (handles YAML parsing of numeric values)
    if isinstance(source_str, int):
        source_str = str(source_str)

    if source_str.startswith("rtsp://") or source_str.startswith("http://"):
        return sv.StreamSource(source_str)
    elif os.path.isfile(source_str):
        return sv.VideoFileSource(source_str)
    elif source_str.isdigit():
        cam_id = int(source_str)
        webcam_kwargs = {"camera_id": cam_id}
        if width is not None:
            webcam_kwargs["width"] = width
        if height is not None:
            webcam_kwargs["height"] = height
        return sv.WebcamSource(**webcam_kwargs)
    else:
        return sv.StreamSource(source_str)


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
        sinks = [sv.DisplaySink(window_name)]
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
        return sv.DisplaySink(window_name)


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
        cap = source.capture.cap if hasattr(source, "capture") else None

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
    except Exception:
        # If anything fails, return defaults
        pass

    # Fallback to reasonable defaults
    return 30, 1280, 720


def get_video_info_with_fallbacks(
    source: Union[sv.WebcamSource, sv.VideoFileSource, sv.StreamSource],
    fallback_fps: int = 30,
    fallback_resolution: tuple = (1920, 1080)
) -> tuple:
    """
    Get video information with configurable fallback values.

    Similar to get_video_info_from_source but allows custom fallback values.
    Validates FPS is in reasonable range (1-120) and resolution is positive.

    Args:
        source: Supervision source
        fallback_fps: FPS to use if unable to read or invalid (default: 30)
        fallback_resolution: (width, height) to use if unable to read (default: 1920x1080)

    Returns:
        Tuple of (fps, width, height)

    Examples:
        >>> source = sv.StreamSource('rtsp://camera/stream')
        >>> fps, w, h = get_video_info_with_fallbacks(source, fallback_fps=60)
    """
    import cv2

    try:
        cap = source.capture.cap if hasattr(source, "capture") else None

        if cap is not None and cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Validate and apply fallbacks
            if fps <= 0 or fps > 120:
                fps = fallback_fps
            if width <= 0 or height <= 0:
                width, height = fallback_resolution

            return int(fps), width, height
    except Exception:
        pass

    return fallback_fps, fallback_resolution[0], fallback_resolution[1]


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


def load_yaml_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary with configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid

    Examples:
        >>> config = load_yaml_config('config.yaml')
        >>> print(config['model'])
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    return config


def parse_point(point_str: str) -> sv.Point:
    """
    Parse a Point from string format 'x,y'.

    Args:
        point_str: String in format "x,y" (e.g., "640,480")

    Returns:
        supervision Point object

    Raises:
        ValueError: If point format is invalid

    Examples:
        >>> parse_point("100,200")
        Point(x=100, y=200)
        >>> parse_point("0,720")
        Point(x=0, y=720)
    """
    try:
        x, y = map(int, point_str.split(","))
        return sv.Point(x=x, y=y)
    except Exception:
        raise ValueError(f"Invalid point format: {point_str}. Expected 'x,y'")


def parse_color(color_str: str) -> sv.Color:
    """
    Parse a Color from string (name or hex code).

    Args:
        color_str: Color name or hex code (e.g., "red", "#FF0000")

    Returns:
        supervision Color object

    Raises:
        ValueError: If color format is invalid

    Supported color names:
        - white, black, red, green, blue, yellow

    Examples:
        >>> parse_color("red")
        Color(r=255, g=0, b=0)
        >>> parse_color("#00FF00")
        Color(r=0, g=255, b=0)
    """
    color_map = {
        "white": sv.Color.WHITE,
        "black": sv.Color.BLACK,
        "red": sv.Color.RED,
        "green": sv.Color.GREEN,
        "blue": sv.Color.BLUE,
        "yellow": sv.Color.YELLOW,
    }

    color_lower = color_str.lower()
    if color_lower in color_map:
        return color_map[color_lower]

    # Try hex color
    try:
        if color_str.startswith("#"):
            color_str = color_str[1:]
        r = int(color_str[0:2], 16)
        g = int(color_str[2:4], 16)
        b = int(color_str[4:6], 16)
        return sv.Color(r=r, g=g, b=b)
    except Exception:
        raise ValueError(
            f"Invalid color: {color_str}. "
            "Use color name (white, black, red, green, blue, yellow) "
            "or hex code (#RRGGBB)"
        )


def create_line_extract_callback():
    """
    Create a callback function to extract line zone counts to data dict.

    This is useful for logging line crossing counts with DebugLoggerStep or
    other metrics collectors.

    Returns:
        Callback function that extracts line_in and line_out from line_zone

    Examples:
        >>> import supervision as sv
        >>> callback = create_line_extract_callback()
        >>> pipeline = (
        ...     sv.Pipeline(source)
        ...     | sv.LineZoneStep(start=start, end=end)
        ...     | sv.CallbackStep(callback)  # Extract counts to data dict
        ...     | sv.DebugLoggerStep(log_file="metrics.json",
        ...                          custom_metrics=["line_in", "line_out"])
        ... )
    """
    def extract_line_counts(data):
        """Extract line crossing counts to data dict."""
        line_zone = data.get("line_zone")
        if line_zone:
            data["line_in"] = line_zone.in_count
            data["line_out"] = line_zone.out_count
        return data

    return extract_line_counts


def create_metrics_overlay_callback(
    position: str = "top-left",
    font_scale: float = 0.5,
    color: tuple = (0, 255, 0),
    bg_opacity: float = 0.6,
    line_spacing: float = 1.5,
    show_fps: bool = True,
    show_detections: bool = True,
    show_tracked: bool = True,
    show_line_counts: bool = True,
    show_roi_info: bool = True,
    show_inference_time: bool = True,
    show_tracking_time: bool = True,
    show_pool_metrics: bool = False,
    frame_counter: Optional[dict] = None,
):
    """
    Create configurable metrics overlay callback.

    Args:
        position: Position of overlay ("top-left", "top-right", "bottom-left", "bottom-right")
        font_scale: Font scale for text (default: 0.5)
        color: Text color in BGR format (default: green (0, 255, 0))
        bg_opacity: Background opacity 0.0-1.0 (default: 0.6)
        line_spacing: Spacing multiplier between lines (default: 1.5)
        show_fps: Show FPS counter
        show_detections: Show number of detections
        show_tracked: Show number of tracked objects
        show_line_counts: Show line crossing counts (if line_zone present)
        show_roi_info: Show ROI dimensions and offset (if ROI present)
        show_inference_time: Show YOLO inference time
        show_tracking_time: Show tracking time
        show_pool_metrics: Show pool detector metrics (if PoolYOLODetectionStep present)
        frame_counter: Dict to track frame count (default: creates new dict)

    Returns:
        Callback function that adds metrics overlay to frame

    Examples:
        >>> # Basic usage
        >>> callback = create_metrics_overlay_callback()
        >>> pipeline = pipeline | sv.CallbackStep(callback)

        >>> # Custom position, color and spacing
        >>> callback = create_metrics_overlay_callback(
        ...     position="top-right",
        ...     color=(255, 255, 0),  # Cyan in BGR
        ...     bg_opacity=0.8,
        ...     line_spacing=2.0  # More space between lines
        ... )

        >>> # Only show specific metrics
        >>> callback = create_metrics_overlay_callback(
        ...     show_fps=True,
        ...     show_detections=True,
        ...     show_tracked=False,
        ...     show_line_counts=True,
        ...     show_roi_info=False
        ... )
    """
    import cv2

    # Initialize frame counter if not provided
    if frame_counter is None:
        frame_counter = {"count": 0}

    def add_metrics_overlay(data):
        """Add metrics text overlay to frame."""
        frame = data.get("frame")
        if frame is None:
            return data

        # Increment frame counter
        frame_counter["count"] += 1

        # Collect metrics
        lines = []

        # Frame count
        lines.append(f"Frame: {frame_counter['count']}")

        # FPS
        if show_fps:
            fps_value = data.get("fps", 0.0)
            lines.append(f"FPS: {fps_value:.1f}")

        # Detections (use all_detections if available, otherwise detections)
        all_detections = data.get("all_detections")
        detections = data.get("detections", sv.Detections.empty())
        if show_detections:
            detection_count = len(all_detections) if all_detections else len(detections)
            lines.append(f"Detections: {detection_count}")

        # Tracked objects
        if show_tracked and detections.tracker_id is not None:
            num_tracked = len(set(detections.tracker_id[detections.tracker_id >= 0]))
            lines.append(f"Tracked: {num_tracked}")

        # Line zone counts
        if show_line_counts:
            line_zone = data.get("line_zone")
            if line_zone is not None:
                lines.append(f"Count IN: {line_zone.in_count}")
                lines.append(f"Count OUT: {line_zone.out_count}")

        # Inference time
        if show_inference_time:
            yolo_metrics = data.get("yolo_metrics", {})
            if yolo_metrics:
                total_time = sum(yolo_metrics.values())
                lines.append(f"Inference: {total_time:.1f}ms")

        # Tracking time
        if show_tracking_time:
            tracker_metrics = data.get("tracker_metrics", {})
            tracking_time = tracker_metrics.get("processing_time_ms", 0.0)
            if tracking_time > 0:
                lines.append(f"Tracking: {tracking_time:.1f}ms")

        # ROI info
        if show_roi_info:
            roi_size = data.get("roi_size")
            roi_offset = data.get("roi_offset")
            if roi_size and roi_offset:
                lines.append(f"ROI: {roi_size[0]}x{roi_size[1]}")
                lines.append(f"ROI Offset: ({roi_offset[0]},{roi_offset[1]})")

        # Pool detector metrics
        if show_pool_metrics:
            pool_detector = data.get("pool_detector")
            if pool_detector is not None:
                metrics = pool_detector.get_metrics()
                queue_current, queue_max = pool_detector.get_queue_size()
                lines.append(f"Workers: {metrics['workers_active']}")
                lines.append(f"Queue: {queue_current}/{queue_max}")
                lines.append(f"Dropped: {metrics['frames_dropped']}")

        # Calculate text dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        padding = 10
        line_height = int(25 * font_scale * line_spacing)

        # Calculate background size
        max_width = 0
        for line in lines:
            (text_width, text_height), _ = cv2.getTextSize(
                line, font, font_scale, thickness
            )
            max_width = max(max_width, text_width)

        bg_width = max_width + 2 * padding
        bg_height = len(lines) * line_height + padding

        # Determine position
        frame_height, frame_width = frame.shape[:2]

        if position == "top-left":
            x_start, y_start = 5, 5
        elif position == "top-right":
            x_start, y_start = frame_width - bg_width - 5, 5
        elif position == "bottom-left":
            x_start, y_start = 5, frame_height - bg_height - 5
        else:  # bottom-right
            x_start, y_start = frame_width - bg_width - 5, frame_height - bg_height - 5

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x_start, y_start),
            (x_start + bg_width, y_start + bg_height),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, bg_opacity, frame, 1 - bg_opacity, 0, frame)

        # Draw text on frame
        y_offset = y_start + line_height
        for line in lines:
            cv2.putText(
                frame,
                line,
                (x_start + padding, y_offset),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            y_offset += line_height

        data["frame"] = frame
        return data

    return add_metrics_overlay


def save_metrics_to_json(metrics_data: list, file_path: str) -> None:
    """
    Save metrics data to a JSON file.

    Args:
        metrics_data: List of dictionaries containing frame metrics
        file_path: Path to output JSON file

    Examples:
        >>> metrics = [
        ...     {'frame': 1, 'fps': 30.5, 'detections': 5},
        ...     {'frame': 2, 'fps': 29.8, 'detections': 3}
        ... ]
        >>> save_metrics_to_json(metrics, 'metrics.json')
    """
    with open(file_path, "w") as f:
        json.dump(metrics_data, f, indent=2)


def plot_metrics_from_json(json_path: str, output_path: str = None) -> str:
    """
    Load metrics from JSON file and create plots.

    Creates a multi-panel plot showing FPS, detections, and inference time
    over the course of the video. Saves the plot as a PNG file with the
    same name as the JSON file.

    Args:
        json_path: Path to JSON file containing metrics
        output_path: Optional custom output path for PNG (default: same as JSON with .png extension)

    Returns:
        Path to the saved plot image

    Examples:
        >>> plot_metrics_from_json('metrics.json')
        'metrics.png'

        >>> plot_metrics_from_json('metrics.json', 'custom_plot.png')
        'custom_plot.png'
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    # Load metrics from JSON
    with open(json_path) as f:
        metrics_data = json.load(f)

    if not metrics_data:
        raise ValueError("No metrics data found in JSON file")

    # Extract data
    frames = [m["frame"] for m in metrics_data]
    fps_values = [m.get("fps", 0) for m in metrics_data]
    detections = [m.get("detections", 0) for m in metrics_data]
    inference_times = [m.get("inference_time_ms", None) for m in metrics_data]
    tracking_times = [m.get("tracking_time_ms", None) for m in metrics_data]

    # Stabilization threshold
    stabilization_frame = 200

    # Get stable data (after stabilization period)
    stable_indices = [i for i, f in enumerate(frames) if f > stabilization_frame]
    stable_fps = [fps_values[i] for i in stable_indices] if stable_indices else fps_values
    stable_detections = [detections[i] for i in stable_indices] if stable_indices else detections

    # Determine number of subplots
    has_inference = any(t is not None for t in inference_times)
    has_tracking = any(t is not None for t in tracking_times)
    num_plots = 2  # FPS + Detections
    if has_inference:
        num_plots += 1
    if has_tracking:
        num_plots += 1

    # Create figure with shared x-axis
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]

    # Plot FPS
    axes[0].plot(frames, fps_values, linewidth=1.5, color="#2E86DE")
    axes[0].set_ylabel("FPS")
    axes[0].set_title("Frames Per Second Over Time")
    axes[0].grid(True, alpha=0.3)

    if stable_fps:
        avg_fps = np.mean(stable_fps)
        min_fps = np.min(stable_fps)
        max_fps = np.max(stable_fps)
        axes[0].axhline(y=avg_fps, color="r", linestyle="--", alpha=0.5, label=f"Avg: {avg_fps:.1f}")
        axes[0].axhline(y=min_fps, color="orange", linestyle="--", alpha=0.5, label=f"Min: {min_fps:.1f}")
        axes[0].axhline(y=max_fps, color="green", linestyle="--", alpha=0.5, label=f"Max: {max_fps:.1f}")

    if len(frames) > stabilization_frame:
        axes[0].axvline(x=stabilization_frame, color="gray", linestyle=":", alpha=0.7, label=f"Stabilization ({stabilization_frame})")

    axes[0].legend()

    # Plot Detections
    axes[1].plot(frames, detections, linewidth=1.5, color="#10AC84")
    axes[1].set_ylabel("Number of Detections")
    axes[1].set_title("Detections Per Frame")
    axes[1].grid(True, alpha=0.3)

    if stable_detections:
        avg_detections = np.mean(stable_detections)
        min_detections = np.min(stable_detections)
        max_detections = np.max(stable_detections)
        axes[1].axhline(y=avg_detections, color="r", linestyle="--", alpha=0.5, label=f"Avg: {avg_detections:.1f}")
        axes[1].axhline(y=min_detections, color="orange", linestyle="--", alpha=0.5, label=f"Min: {min_detections}")
        axes[1].axhline(y=max_detections, color="green", linestyle="--", alpha=0.5, label=f"Max: {max_detections}")

    if len(frames) > stabilization_frame:
        axes[1].axvline(x=stabilization_frame, color="gray", linestyle=":", alpha=0.7, label=f"Stabilization ({stabilization_frame})")

    axes[1].legend()

    # Plot Inference Time if available
    current_plot_idx = 2
    if has_inference:
        valid_inference = [(f, t) for f, t in zip(frames, inference_times) if t is not None]
        if valid_inference:
            inf_frames, inf_times = zip(*valid_inference)
            axes[current_plot_idx].plot(inf_frames, inf_times, linewidth=1.5, color="#EE5A6F")
            axes[current_plot_idx].set_ylabel("Inference Time (ms)")
            axes[current_plot_idx].set_title("Inference Time Per Frame")
            axes[current_plot_idx].grid(True, alpha=0.3)

            stable_inference_indices = [i for i, f in enumerate(inf_frames) if f > stabilization_frame]
            stable_inference = [inf_times[i] for i in stable_inference_indices] if stable_inference_indices else inf_times

            if stable_inference:
                avg_inference = np.mean(stable_inference)
                min_inference = np.min(stable_inference)
                max_inference = np.max(stable_inference)
                axes[current_plot_idx].axhline(y=avg_inference, color="r", linestyle="--", alpha=0.5, label=f"Avg: {avg_inference:.1f}ms")
                axes[current_plot_idx].axhline(y=min_inference, color="orange", linestyle="--", alpha=0.5, label=f"Min: {min_inference:.1f}ms")
                axes[current_plot_idx].axhline(y=max_inference, color="green", linestyle="--", alpha=0.5, label=f"Max: {max_inference:.1f}ms")

            if len(inf_frames) > stabilization_frame:
                axes[current_plot_idx].axvline(x=stabilization_frame, color="gray", linestyle=":", alpha=0.7, label=f"Stabilization ({stabilization_frame})")

            axes[current_plot_idx].legend()
        current_plot_idx += 1

    # Plot Tracking Time if available
    if has_tracking:
        valid_tracking = [(f, t) for f, t in zip(frames, tracking_times) if t is not None]
        if valid_tracking:
            track_frames, track_times = zip(*valid_tracking)
            axes[current_plot_idx].plot(track_frames, track_times, linewidth=1.5, color="#A55EEA")
            axes[current_plot_idx].set_ylabel("Tracking Time (ms)")
            axes[current_plot_idx].set_title("Tracking Time Per Frame")
            axes[current_plot_idx].grid(True, alpha=0.3)

            stable_tracking_indices = [i for i, f in enumerate(track_frames) if f > stabilization_frame]
            stable_tracking = [track_times[i] for i in stable_tracking_indices] if stable_tracking_indices else track_times

            if stable_tracking:
                avg_tracking = np.mean(stable_tracking)
                min_tracking = np.min(stable_tracking)
                max_tracking = np.max(stable_tracking)
                axes[current_plot_idx].axhline(y=avg_tracking, color="r", linestyle="--", alpha=0.5, label=f"Avg: {avg_tracking:.1f}ms")
                axes[current_plot_idx].axhline(y=min_tracking, color="orange", linestyle="--", alpha=0.5, label=f"Min: {min_tracking:.1f}ms")
                axes[current_plot_idx].axhline(y=max_tracking, color="green", linestyle="--", alpha=0.5, label=f"Max: {max_tracking:.1f}ms")

            if len(track_frames) > stabilization_frame:
                axes[current_plot_idx].axvline(x=stabilization_frame, color="gray", linestyle=":", alpha=0.7, label=f"Stabilization ({stabilization_frame})")

            axes[current_plot_idx].legend()

    # Add x-label to the last subplot
    axes[-1].set_xlabel("Frame")

    plt.tight_layout()

    # Determine output path
    if output_path is None:
        json_file = Path(json_path)
        output_path = json_file.with_suffix(".png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)
