"""
Plot Debug Metrics

This script reads metrics logged by DebugLoggerStep and creates visualizations
of critical performance metrics.

The script generates plots for:
- FPS over time
- Detection counts per frame
- Detection time/latency metrics
- Tracking metrics (if available)
- Class distribution
- Confidence scores

Usage:
    # Basic usage
    python plot_debug_metrics.py debug_metrics.json

    # Save plots to file instead of displaying
    python plot_debug_metrics.py debug_metrics.json --output plots.png

    # Only show specific metrics
    python plot_debug_metrics.py debug_metrics.json --metrics fps detections

    # Custom figure size
    python plot_debug_metrics.py debug_metrics.json --figsize 16 10
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(log_file: str) -> list[dict]:
    """Load metrics from JSON log file."""
    with open(log_file, "r") as f:
        return json.load(f)


def plot_fps(ax, metrics: list[dict]) -> None:
    """Plot FPS over time."""
    frames = [m["frame_number"] for m in metrics if "fps" in m]
    fps = [m["fps"] for m in metrics if "fps" in m]

    if not fps:
        ax.text(0.5, 0.5, "No FPS data available", ha="center", va="center")
        ax.set_title("FPS Over Time")
        return

    ax.plot(frames, fps, linewidth=1, alpha=0.7)
    ax.axhline(np.mean(fps), color="r", linestyle="--", label=f"Mean: {np.mean(fps):.1f}")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("FPS")
    ax.set_title("FPS Over Time")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_detections(ax, metrics: list[dict]) -> None:
    """Plot detection counts over time."""
    frames = [m["frame_number"] for m in metrics if "detection_count" in m]
    counts = [m["detection_count"] for m in metrics if "detection_count" in m]

    if not counts:
        ax.text(0.5, 0.5, "No detection data available", ha="center", va="center")
        ax.set_title("Detection Count Over Time")
        return

    ax.plot(frames, counts, linewidth=1, alpha=0.7)
    ax.axhline(np.mean(counts), color="r", linestyle="--", label=f"Mean: {np.mean(counts):.1f}")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Number of Detections")
    ax.set_title("Detection Count Over Time")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_timing_metrics(ax, metrics: list[dict]) -> None:
    """Plot timing metrics (detection time, inference time, etc.)."""
    timing_keys = [
        "avg_inference_time_ms",
        "avg_queue_time_ms",
        "detection_time_ms",
        "tracking_time_ms",
    ]

    # Find which timing metrics are available
    available_metrics = {}
    for key in timing_keys:
        values = [m[key] for m in metrics if key in m]
        if values:
            available_metrics[key] = values

    if not available_metrics:
        ax.text(0.5, 0.5, "No timing data available", ha="center", va="center")
        ax.set_title("Timing Metrics")
        return

    # Plot each available metric
    frames = [m["frame_number"] for m in metrics]
    for key, values in available_metrics.items():
        # Pad values to match frame count
        frames_for_key = [m["frame_number"] for m in metrics if key in m]
        label = key.replace("_", " ").replace("ms", "(ms)").title()
        ax.plot(frames_for_key, values, label=label, linewidth=1, alpha=0.7)

    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Processing Time Metrics")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_tracking_metrics(ax, metrics: list[dict]) -> None:
    """Plot tracking metrics (tracked objects, unique IDs)."""
    frames = [m["frame_number"] for m in metrics if "tracked_count" in m]
    tracked = [m["tracked_count"] for m in metrics if "tracked_count" in m]
    unique_ids = [m["unique_tracker_ids"] for m in metrics if "unique_tracker_ids" in m]

    if not tracked and not unique_ids:
        ax.text(0.5, 0.5, "No tracking data available", ha="center", va="center")
        ax.set_title("Tracking Metrics")
        return

    if tracked:
        ax.plot(frames, tracked, label="Tracked Objects", linewidth=1, alpha=0.7)

    if unique_ids:
        frames_ids = [m["frame_number"] for m in metrics if "unique_tracker_ids" in m]
        ax.plot(frames_ids, unique_ids, label="Unique Tracker IDs", linewidth=1, alpha=0.7)

    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Count")
    ax.set_title("Tracking Metrics")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_confidence_stats(ax, metrics: list[dict]) -> None:
    """Plot confidence statistics over time."""
    frames = [m["frame_number"] for m in metrics if "confidence_mean" in m]
    conf_mean = [m["confidence_mean"] for m in metrics if "confidence_mean" in m]
    conf_min = [m["confidence_min"] for m in metrics if "confidence_min" in m]
    conf_max = [m["confidence_max"] for m in metrics if "confidence_max" in m]

    if not conf_mean:
        ax.text(0.5, 0.5, "No confidence data available", ha="center", va="center")
        ax.set_title("Confidence Statistics")
        return

    ax.plot(frames, conf_mean, label="Mean Confidence", linewidth=1, alpha=0.7)
    ax.fill_between(frames, conf_min, conf_max, alpha=0.2, label="Min-Max Range")

    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Confidence")
    ax.set_title("Detection Confidence Statistics")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_class_distribution(ax, metrics: list[dict]) -> None:
    """Plot overall class distribution."""
    # Aggregate class counts across all frames
    class_totals = {}
    for m in metrics:
        if "class_distribution" in m:
            for class_id, count in m["class_distribution"].items():
                class_totals[int(class_id)] = class_totals.get(int(class_id), 0) + count

    if not class_totals:
        ax.text(0.5, 0.5, "No class distribution data", ha="center", va="center")
        ax.set_title("Class Distribution")
        return

    # Sort by class ID
    class_ids = sorted(class_totals.keys())
    counts = [class_totals[cid] for cid in class_ids]

    ax.bar(class_ids, counts, alpha=0.7)
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Total Detections")
    ax.set_title("Class Distribution (Total)")
    ax.grid(True, alpha=0.3, axis="y")


def create_summary_text(metrics: list[dict]) -> str:
    """Create summary text with key statistics."""
    if not metrics:
        return "No metrics available"

    lines = ["Summary Statistics:", "=" * 40]

    # FPS stats
    fps_values = [m["fps"] for m in metrics if "fps" in m]
    if fps_values:
        lines.append(f"FPS: {np.mean(fps_values):.1f} ± {np.std(fps_values):.1f}")
        lines.append(f"  Min: {np.min(fps_values):.1f}, Max: {np.max(fps_values):.1f}")

    # Detection stats
    det_counts = [m["detection_count"] for m in metrics if "detection_count" in m]
    if det_counts:
        lines.append(f"Detections per frame: {np.mean(det_counts):.1f} ± {np.std(det_counts):.1f}")
        lines.append(f"  Min: {np.min(det_counts)}, Max: {np.max(det_counts)}")
        lines.append(f"  Total detections: {sum(det_counts)}")

    # Timing stats
    inf_times = [m["avg_inference_time_ms"] for m in metrics if "avg_inference_time_ms" in m]
    if inf_times:
        lines.append(f"Avg inference time: {np.mean(inf_times):.1f} ms")

    # Frame count
    lines.append(f"Total frames: {len(metrics)}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Plot debug metrics from DebugLoggerStep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "log_file",
        type=str,
        help="Path to JSON log file from DebugLoggerStep",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Save plot to file instead of displaying (e.g., plots.png)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["fps", "detections", "timing", "tracking", "confidence", "classes"],
        default=None,
        help="Specific metrics to plot (default: all available)",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=int,
        default=[15, 10],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 15 10)",
    )

    args = parser.parse_args()

    # Load metrics
    print(f"Loading metrics from: {args.log_file}")
    metrics = load_metrics(args.log_file)
    print(f"Loaded {len(metrics)} frame metrics")

    if not metrics:
        print("No metrics found in log file")
        return

    # Print summary
    print("\n" + create_summary_text(metrics))
    print()

    # Determine which metrics to plot
    plot_all = args.metrics is None
    plot_map = {
        "fps": plot_fps,
        "detections": plot_detections,
        "timing": plot_timing_metrics,
        "tracking": plot_tracking_metrics,
        "confidence": plot_confidence_stats,
        "classes": plot_class_distribution,
    }

    if plot_all:
        selected_plots = list(plot_map.items())
    else:
        selected_plots = [(name, func) for name, func in plot_map.items() if name in args.metrics]

    # Create figure with subplots
    n_plots = len(selected_plots)
    if n_plots == 0:
        print("No metrics selected for plotting")
        return

    # Arrange in 2 columns
    n_rows = (n_plots + 1) // 2
    n_cols = min(2, n_plots)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=tuple(args.figsize))
    fig.suptitle(f"Debug Metrics: {Path(args.log_file).name}", fontsize=16)

    # Flatten axes for easier indexing
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    # Plot each metric
    for idx, (name, plot_func) in enumerate(selected_plots):
        print(f"Plotting {name}...")
        plot_func(axes[idx], metrics)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    # Save or display
    if args.output:
        print(f"Saving plot to: {args.output}")
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print("Done!")
    else:
        print("Displaying plot (close window to exit)...")
        plt.show()


if __name__ == "__main__":
    main()
