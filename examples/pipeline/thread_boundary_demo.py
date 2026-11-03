"""
ThreadBoundary demo: the same detection + tracking pipeline with and without
the boundary, its frame rate, and a check that the output is identical.

With the boundary, the source and YOLO run in a worker thread one frame ahead
of ByteTrack, which runs in the calling thread. The gain shows when inference
(GPU) and tracking (CPU) take comparable time.

Usage:
    python examples/pipeline/thread_boundary_demo.py --input video.mp4 \
        --model yolov8n.pt --frames 1000

    # Also write both runs annotated (boxes, tracker ids, traces):
    # tracked_1thread.mp4 and tracked_2threads.mp4
    python examples/pipeline/thread_boundary_demo.py --input video.mp4 \
        --output tracked.mp4
"""

import argparse
import time
from pathlib import Path

import supervision as sv


def run(args: argparse.Namespace, boundary: bool) -> tuple[float, list]:
    yolo_step = sv.YOLODetectionStep(args.model, conf=args.conf, verbose=False)
    pipeline = sv.Pipeline(sv.VideoFileSource(args.input)) | yolo_step
    if boundary:
        pipeline = pipeline | sv.ThreadBoundary()
    pipeline = pipeline | sv.ByteTrackerStep()
    if args.output:
        out = Path(args.output)
        suffix = "_2threads" if boundary else "_1thread"
        info = sv.VideoInfo.from_video_path(args.input)
        pipeline = (
            pipeline
            | sv.TrackerAnnotatorStep(
                class_names=yolo_step.model.names,
                trace_length=50,
                trace_thickness=2,
                copy_frame=False,
            )
            | sv.VideoFileSink(
                str(out.with_stem(out.stem + suffix)),
                fps=info.fps,
                width=info.width,
                height=info.height,
            )
        )

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
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence")
    parser.add_argument("--frames", type=int, default=1000, help="Frames to run")
    parser.add_argument("--output", help="Annotated video path (both runs)")
    args = parser.parse_args()

    fps_one, ids_one = run(args, boundary=False)
    fps_two, ids_two = run(args, boundary=True)
    print(f"one thread:  {fps_one:6.1f} fps")
    print(f"two threads: {fps_two:6.1f} fps")
    print("same tracker ids:", ids_one == ids_two)


if __name__ == "__main__":
    main()
