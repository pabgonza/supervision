"""
Asynchronous YOLO detection demo with strategy comparison.

This example demonstrates how to use AsyncYOLODetectionStep with different
strategies to handle scenarios where the detector is slower than the frame rate.

Requirements:
    pip install ultralytics

Usage:
    # Test with webcam (default: USE_LAST_RESULT strategy)
    python examples/pipeline/async_detection_demo.py --model yolov8n.pt

    # Test with specific strategy
    python examples/pipeline/async_detection_demo.py --model yolov8n.pt \
        --strategy skip

    # Test with video file
    python examples/pipeline/async_detection_demo.py --model yolov8n.pt \
        --source file --input video.mp4 --strategy cache

    # Test with RTSP stream
    python examples/pipeline/async_detection_demo.py --model yolov8n.pt \
        --source stream --input rtsp://camera.local/stream

    # With output
    python examples/pipeline/async_detection_demo.py --model yolov8n.pt \
        --source stream --input rtsp://camera.local/stream --output detections.mp4

    # Compare all strategies side-by-side (webcam)
    python examples/pipeline/async_detection_demo.py --model yolov8n.pt \
        --compare

    # Compare all strategies with stream
    python examples/pipeline/async_detection_demo.py --model yolov8n.pt \
        --compare --source stream --input rtsp://192.168.1.100/stream

Strategies:
    skip  : Skip frames when detector is busy (lowest latency)
    cache : Always return cached result while processing (balanced)
    queue : Queue frames for processing (best accuracy, higher latency)
    sync  : Traditional synchronous processing (baseline)
"""

import argparse
import time

import cv2
import numpy as np
import supervision as sv

import utils


def create_metrics_overlay_callback(detector):
    """
    Create a callback function that overlays detector metrics on the frame.

    Args:
        detector: AsyncYOLODetectionStep instance to get metrics from

    Returns:
        Callback function for use with CallbackStep
    """

    def overlay_metrics(data: dict) -> None:
        """Draw metrics overlay on frame."""
        frame = data.get("frame")
        if frame is None:
            return

        # Get metrics
        metrics = detector.get_metrics()
        queue_size, max_queue = detector.get_queue_size()
        fps = data.get("fps", 0)

        # Prepare metrics text
        metrics_text = [
            f"FPS: {fps:.1f}",
            f"Processed: {metrics['frames_processed']}",
            f"Cached: {metrics['frames_cached']}",
            f"Skipped: {metrics['frames_skipped']}",
            f"Queue: {queue_size}/{max_queue}",
            f"Queue Full: {metrics['queue_full_count']}",
            f"Inference: {metrics['avg_inference_time_ms']:.1f}ms",
        ]

        # Position and styling
        x, y = 10, 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        line_height = 30
        text_color = (255, 255, 255)  # White
        bg_color = (0, 0, 0)  # Black

        # Draw semi-transparent background
        overlay = frame.copy()
        bg_height = len(metrics_text) * line_height + 20
        cv2.rectangle(overlay, (5, 5), (300, bg_height), bg_color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw text
        for i, text in enumerate(metrics_text):
            y_pos = y + i * line_height
            cv2.putText(
                frame, text, (x, y_pos), font, font_scale, text_color, font_thickness
            )

    return overlay_metrics


def run_single_strategy(
    model_path: str,
    strategy: sv.DetectionStrategy,
    source: str = "webcam",
    input_path: str | None = None,
    camera_id: int = 0,
    conf: float = 0.25,
    device: str = "cuda",
    output_path: str | None = None,
):
    """
    Run async detection with a single strategy.

    Args:
        model_path: Path to YOLO model
        strategy: Detection strategy to use
        source: Source type ('webcam', 'file', or 'stream')
        input_path: Path to video file or stream URL (for file/stream source)
        camera_id: Camera ID (for webcam source)
        conf: Detection confidence threshold
        device: Device for inference
        output_path: Optional path to save output video
    """
    strategy_name = strategy.value.upper()
    print(f"\n=== Running with {strategy_name} strategy ===")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Source: {source}")
    print("Press 'q' to quit\n")

    # Create detector with selected strategy
    detector = sv.AsyncYOLODetectionStep(
        model_path=model_path,
        conf=conf,
        device=device,
        strategy=strategy,
        max_queue_size=10 if strategy == sv.DetectionStrategy.QUEUE_LATEST else 1,
        warmup=True,
    )

    # Create source using utils
    from argparse import Namespace
    args_for_source = Namespace(
        source=source,
        input=input_path,
        camera=camera_id
    )
    pipeline_source = utils.create_source_from_args(args_for_source)

    # Build pipeline with metrics overlay
    pipeline = (
        sv.Pipeline(pipeline_source)
        | sv.FPSCalculatorStep()
        | detector
        | sv.BoxAnnotatorStep(copy_frame=False)
        | sv.LabelAnnotatorStep(copy_frame=False)
        | sv.CallbackStep(create_metrics_overlay_callback(detector))
    )

    # Add sink(s)
    if output_path:
        pipeline = pipeline | sv.MultiSink([
            sv.DisplaySink(f"Async Detection - {strategy_name}", show_fps=False),
            sv.VideoFileSink(output_path)
        ])
    else:
        pipeline = pipeline | sv.DisplaySink(f"Async Detection - {strategy_name}", show_fps=False)

    try:
        for data in pipeline:
            # Metrics are now displayed on frame via CallbackStep
            pass

    except (KeyboardInterrupt, StopIteration):
        print(f"\n\nStopped {strategy_name} detection")

    finally:
        # Print final metrics
        metrics = detector.get_metrics()
        print(f"\n=== Final Metrics ({strategy_name}) ===")
        print(f"Total frames processed: {metrics['frames_processed']}")
        print(f"Total frames cached: {metrics['frames_cached']}")
        print(f"Total frames skipped: {metrics['frames_skipped']}")
        print(f"Total frames queued: {metrics['frames_queued']}")
        print(f"Queue full count: {metrics['queue_full_count']}")
        print(f"Avg inference time: {metrics['avg_inference_time_ms']:.1f}ms")

        # Stop detector
        detector.stop()


def compare_strategies(
    model_path: str,
    source: str = "webcam",
    input_path: str | None = None,
    camera_id: int = 0,
    conf: float = 0.25,
    device: str = "cuda",
):
    """
    Compare all strategies side-by-side in separate windows.

    Args:
        model_path: Path to YOLO model
        source: Source type ('webcam', 'file', or 'stream')
        input_path: Path to video file or stream URL (for file/stream source)
        camera_id: Camera ID (for webcam source)
        conf: Detection confidence threshold
        device: Device for inference
    """
    print("\n=== Comparing All Strategies ===")
    print("This will open 4 windows showing different strategies")
    print("Press 'q' in any window to quit\n")

    # Create detectors with different strategies
    strategies = [
        (sv.DetectionStrategy.SKIP_WHEN_BUSY, "SKIP"),
        (sv.DetectionStrategy.USE_LAST_RESULT, "CACHE"),
        (sv.DetectionStrategy.QUEUE_LATEST, "QUEUE"),
        (sv.DetectionStrategy.SYNCHRONOUS, "SYNC"),
    ]

    detectors = []
    for strategy, name in strategies:
        detector = sv.AsyncYOLODetectionStep(
            model_path=model_path,
            conf=conf,
            device=device,
            strategy=strategy,
            max_queue_size=2 if strategy == sv.DetectionStrategy.QUEUE_LATEST else 1,
            warmup=(len(detectors) == 0),  # Only warmup first detector
        )
        detectors.append((detector, name))

    # Create source using utils
    from argparse import Namespace
    args_for_source = Namespace(
        source=source,
        input=input_path,
        camera=camera_id
    )
    pipeline_source = utils.create_source_from_args(args_for_source)

    # Process frames manually to show in multiple windows
    try:
        frame_count = 0
        for source_data in pipeline_source:
            frame_count += 1
            frame = source_data["frame"]

            # Process frame with each detector
            for detector, name in detectors:
                # Create data dict
                data = {"frame": frame.copy()}

                # Process through detector
                data = detector.process(data)

                # Annotate
                annotated = data["frame"].copy()
                if "detections" in data and len(data["detections"]) > 0:
                    annotated = sv.BoxAnnotator().annotate(
                        scene=annotated, detections=data["detections"]
                    )

                # Add strategy label
                cv2.putText(
                    annotated,
                    f"Strategy: {name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

                # Show metrics
                metrics = detector.get_metrics()
                queue_size, max_queue = detector.get_queue_size()
                y_offset = 70
                metrics_text = [
                    f"Processed: {metrics['frames_processed']}",
                    f"Cached: {metrics['frames_cached']}",
                    f"Skipped: {metrics['frames_skipped']}",
                    f"Queue: {queue_size}/{max_queue}",
                    f"Queue Full: {metrics['queue_full_count']}",
                    f"Inference: {metrics['avg_inference_time_ms']:.1f}ms",
                ]

                for i, text in enumerate(metrics_text):
                    cv2.putText(
                        annotated,
                        text,
                        (10, y_offset + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                # Display
                cv2.imshow(f"Async Detection - {name}", annotated)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except (KeyboardInterrupt, StopIteration):
        print("\n\nStopped comparison")

    finally:
        # Cleanup
        cv2.destroyAllWindows()
        for detector, name in detectors:
            detector.stop()

        # Print final comparison
        print("\n=== Final Comparison ===")
        for detector, name in detectors:
            metrics = detector.get_metrics()
            print(f"\n{name}:")
            print(f"  Processed: {metrics['frames_processed']}")
            print(f"  Cached: {metrics['frames_cached']}")
            print(f"  Skipped: {metrics['frames_skipped']}")
            print(f"  Queue full count: {metrics['queue_full_count']}")
            print(f"  Avg inference: {metrics['avg_inference_time_ms']:.1f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="Async YOLO detection with strategy comparison"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO model file (.pt)",
    )

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
        help="Camera ID (default: 0)",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["skip", "cache", "queue", "sync"],
        default="cache",
        help="Detection strategy (default: cache)",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all strategies side-by-side",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video file path (optional, only for single strategy mode)",
    )

    args = parser.parse_args()

    # Map strategy string to enum
    strategy_map = {
        "skip": sv.DetectionStrategy.SKIP_WHEN_BUSY,
        "cache": sv.DetectionStrategy.USE_LAST_RESULT,
        "queue": sv.DetectionStrategy.QUEUE_LATEST,
        "sync": sv.DetectionStrategy.SYNCHRONOUS,
    }

    if args.compare:
        compare_strategies(
            model_path=args.model,
            source=args.source,
            input_path=args.input,
            camera_id=args.camera,
            conf=args.conf,
            device=args.device,
        )
    else:
        strategy = strategy_map[args.strategy]
        run_single_strategy(
            model_path=args.model,
            strategy=strategy,
            source=args.source,
            input_path=args.input,
            camera_id=args.camera,
            conf=args.conf,
            device=args.device,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
