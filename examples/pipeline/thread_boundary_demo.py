"""
ThreadBoundary demo: the same detection + tracking pipeline with and without
the boundary, its frame rate, and a check that the output is identical.

With the boundary, the source and YOLO run in a worker thread one frame ahead
of ByteTrack, which runs in the calling thread. The gain shows when inference
(GPU) and tracking (CPU) take comparable time.

Usage:
    python examples/pipeline/thread_boundary_demo.py --input video.mp4 \
        --model yolov8n.pt --frames 1000
"""

import argparse
import time

import supervision as sv


def run(args: argparse.Namespace, boundary: bool) -> tuple[float, list]:
    pipeline = sv.Pipeline(sv.VideoFileSource(args.input)) | sv.YOLODetectionStep(
        args.model
    )
    if boundary:
        pipeline = pipeline | sv.ThreadBoundary()
    pipeline = pipeline | sv.ByteTrackerStep()

    ids, start = [], time.perf_counter()
    for n, data in enumerate(pipeline, 1):
        ids.append(data["detections"].tracker_id.tolist())
        if n == args.frames:
            break
    return len(ids) / (time.perf_counter() - start), ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, help="Video file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--frames", type=int, default=1000, help="Frames to run")
    args = parser.parse_args()

    fps_one, ids_one = run(args, boundary=False)
    fps_two, ids_two = run(args, boundary=True)
    print(f"one thread:  {fps_one:6.1f} fps")
    print(f"two threads: {fps_two:6.1f} fps")
    print("same tracker ids:", ids_one == ids_two)


if __name__ == "__main__":
    main()
