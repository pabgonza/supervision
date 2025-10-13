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
        --input video.mp4 --strategy cache

    # Compare all strategies side-by-side
    python examples/pipeline/async_detection_demo.py --model yolov8n.pt \
        --compare

Strategies:
    skip  : Skip frames when detector is busy (lowest latency)
    cache : Always return cached result while processing (balanced)
    queue : Queue frames for processing (best accuracy, higher latency)
    sync  : Traditional synchronous processing (baseline)
"""

import argparse
import time

import cv2
import supervision as sv


def run_single_strategy(
    model_path: str,
    strategy: sv.DetectionStrategy,
    source: str = "webcam",
    video_path: str | None = None,
    camera_id: int = 0,
    conf: float = 0.25,
    device: str = "cuda",
):
    """
    Run async detection with a single strategy.

    Args:
        model_path: Path to YOLO model
        strategy: Detection strategy to use
        source: Source type ('webcam' or 'file')
        video_path: Path to video file (for file source)
        camera_id: Camera ID (for webcam source)
        conf: Detection confidence threshold
        device: Device for inference
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
        max_queue_size=2 if strategy == sv.DetectionStrategy.QUEUE_LATEST else 1,
        warmup=True,
    )

    # Create source
    if source == "webcam":
        pipeline_source = sv.WebcamSource(camera_id=camera_id)
    else:
        if not video_path:
            raise ValueError("--input required for file source")
        pipeline_source = sv.VideoFileSource(video_path)

    # Build pipeline
    pipeline = (
        sv.Pipeline(pipeline_source)
        | sv.FPSCalculatorStep()
        | detector
        | sv.BoxAnnotatorStep()
        | sv.DisplaySink(f"Async Detection - {strategy_name}", show_fps=True)
    )

    # Metrics display
    last_metrics_time = time.time()
    metrics_interval = 2.0  # Update metrics every 2 seconds

    try:
        for data in pipeline:
            # Periodically print metrics
            current_time = time.time()
            if current_time - last_metrics_time >= metrics_interval:
                metrics = detector.get_metrics()
                fps = data.get("fps", 0)

                print(f"\n--- Metrics ({strategy_name}) ---")
                print(f"Pipeline FPS: {fps:.1f}")
                print(f"Frames processed: {metrics['frames_processed']}")
                print(f"Frames cached: {metrics['frames_cached']}")
                print(f"Frames skipped: {metrics['frames_skipped']}")
                print(f"Frames queued: {metrics['frames_queued']}")
                print(f"Avg inference time: {metrics['avg_inference_time_ms']:.1f}ms")
                print(f"Queue full count: {metrics['queue_full_count']}")

                last_metrics_time = current_time

    except (KeyboardInterrupt, StopIteration):
        print(f"\n\nStopped {strategy_name} detection")

    finally:
        # Print final metrics
        metrics = detector.get_metrics()
        print(f"\n=== Final Metrics ({strategy_name}) ===")
        print(f"Total frames processed: {metrics['frames_processed']}")
        print(f"Total frames cached: {metrics['frames_cached']}")
        print(f"Total frames skipped: {metrics['frames_skipped']}")
        print(f"Avg inference time: {metrics['avg_inference_time_ms']:.1f}ms")

        # Stop detector
        detector.stop()


def compare_strategies(
    model_path: str,
    source: str = "webcam",
    video_path: str | None = None,
    camera_id: int = 0,
    conf: float = 0.25,
    device: str = "cuda",
):
    """
    Compare all strategies side-by-side in separate windows.

    Args:
        model_path: Path to YOLO model
        source: Source type ('webcam' or 'file')
        video_path: Path to video file (for file source)
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

    # Create source
    if source == "webcam":
        pipeline_source = sv.WebcamSource(camera_id=camera_id)
    else:
        if not video_path:
            raise ValueError("--input required for file source")
        pipeline_source = sv.VideoFileSource(video_path)

    # Process frames manually to show in multiple windows
    try:
        frame_count = 0
        for frame in pipeline_source:
            frame_count += 1

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
                y_offset = 70
                metrics_text = [
                    f"Processed: {metrics['frames_processed']}",
                    f"Cached: {metrics['frames_cached']}",
                    f"Skipped: {metrics['frames_skipped']}",
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
        choices=["webcam", "file"],
        default="webcam",
        help="Source type (default: webcam)",
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Path to video file (for file source)",
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
            video_path=args.input,
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
            video_path=args.input,
            camera_id=args.camera,
            conf=args.conf,
            device=args.device,
        )


if __name__ == "__main__":
    main()
