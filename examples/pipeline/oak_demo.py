"""
Luxonis OAK camera demo for supervision pipelines.

This demonstrates the three OAK sources:
- OakRgbSource: RGB camera only
- OakStereoSource: Multiple stereo streams
- OakRgbDepthSource: Aligned RGB + Depth

Usage:
    # RGB only (OAK-1 or OAK-D RGB sensor)
    python examples/pipeline/oak_demo.py --mode rgb

    # Stereo depth (OAK-D)
    python examples/pipeline/oak_demo.py --mode stereo

    # RGB + aligned depth (OAK-D)
    python examples/pipeline/oak_demo.py --mode rgb_depth

    # With specific device
    python examples/pipeline/oak_demo.py --mode rgb --device-mxid 14442C10D13EAFD000

    # Custom resolution
    python examples/pipeline/oak_demo.py --mode rgb --width 1920 --height 1080 --fps 30

Press 'q' or ESC to stop.
"""

import argparse

import cv2
import numpy as np

import supervision as sv


def main():
    parser = argparse.ArgumentParser(description="Luxonis OAK camera demo")

    parser.add_argument(
        "--mode",
        type=str,
        default="rgb",
        choices=["rgb", "stereo", "rgb_depth"],
        help="Camera mode: rgb, stereo, or rgb_depth",
    )
    parser.add_argument(
        "--device-mxid",
        type=str,
        default=None,
        help="Device MxId (auto-detect if not specified)",
    )
    parser.add_argument("--width", type=int, default=1920, help="RGB frame width")
    parser.add_argument("--height", type=int, default=1080, help="RGB frame height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument(
        "--depth-resolution",
        type=str,
        default="400p",
        choices=["400p", "480p", "720p", "800p"],
        help="Depth resolution",
    )

    args = parser.parse_args()

    print("Luxonis OAK Camera Demo")
    print("-" * 40)
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device_mxid or 'auto-detect'}")
    print("-" * 40)

    if args.mode == "rgb":
        source = sv.OakRgbSource(
            device_mxid=args.device_mxid,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )

        pipeline = sv.Pipeline(source) | sv.FPSCalculatorStep()

        for data in pipeline:
            frame = data["frame"]
            fps = data.get("fps", 0)

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("OAK RGB", frame)
            if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                break

    elif args.mode == "stereo":
        source = sv.OakStereoSource(
            device_mxid=args.device_mxid,
            output_streams=["left", "right", "depth", "disparity"],
            resolution=args.depth_resolution,
            extended_disparity=True,
        )

        pipeline = sv.Pipeline(source)

        for data in pipeline:
            streams = data["streams"]

            depth = streams["depth"]
            depth_normalized = cv2.normalize(
                depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

            disparity = streams["disparity"]
            disparity_normalized = cv2.normalize(
                disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )

            cv2.imshow("Left", streams["left"])
            cv2.imshow("Right", streams["right"])
            cv2.imshow("Depth", depth_colored)
            cv2.imshow("Disparity", disparity_normalized)

            if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                break

    elif args.mode == "rgb_depth":
        source = sv.OakRgbDepthSource(
            device_mxid=args.device_mxid,
            rgb_width=args.width,
            rgb_height=args.height,
            depth_resolution=args.depth_resolution,
            fps=args.fps,
            extended_disparity=True,
            align_to_rgb=True,
        )

        pipeline = sv.Pipeline(source) | sv.FPSCalculatorStep()

        for data in pipeline:
            rgb = data["frame"]
            depth = data["depth"]
            fps = data.get("fps", 0)

            depth_normalized = cv2.normalize(
                depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

            if depth_colored.shape[:2] != rgb.shape[:2]:
                depth_colored = cv2.resize(depth_colored, (rgb.shape[1], rgb.shape[0]))

            combined = np.hstack([rgb, depth_colored])

            cv2.putText(
                combined,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("RGB + Depth", combined)
            if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                break

    cv2.destroyAllWindows()
    pipeline.close()
    print("\nDemo stopped.")


if __name__ == "__main__":
    main()
