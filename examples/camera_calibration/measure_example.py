"""
Measurement Example

This script demonstrates how to measure real-world distances in images
after camera calibration.

Usage:
    python measure_example.py --camera_coefs camera_calibration.json --image test.jpg --distance 500
    python measure_example.py --camera_coefs camera_calibration.json --image test.jpg --reference-real 24.4
"""

import argparse
import cv2
import numpy as np

from calibrate_camera import (
    CameraCoefficients,
    undistort_image,
    calculate_pixel_size,
    pixels_to_real_size,
    calculate_scale_from_reference,
    estimate_distance_from_known_object
)


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate measurement from calibrated camera images"
    )
    parser.add_argument(
        "--camera_coefs",
        type=str,
        required=True,
        help="Path to camera calibration JSON file"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image to measure"
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="Known distance from camera to object plane (in same units as square_size)"
    )
    parser.add_argument(
        "--reference-real",
        type=float,
        default=None,
        help="Known real size of reference object (will prompt to click two points to measure it)"
    )

    args = parser.parse_args()

    # Load calibration
    print(f"Loading calibration from: {args.camera_coefs}")
    coeffs = CameraCoefficients.load(args.camera_coefs)
    print(f"\n{coeffs}\n")

    # Load image
    print(f"Loading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image {args.image}")
        return

    # Undistort image
    print("Undistorting image...")
    undistorted = undistort_image(image, coeffs, crop=False, balance=1.0)

    # Display measurement methods
    print("\n" + "="*60)
    print("MEASUREMENT METHODS")
    print("="*60)

    # Method 1: Known distance
    if args.distance is not None:
        print(f"\nMethod 1: Using known distance ({args.distance} units)")
        print("-" * 60)

        pixel_w, pixel_h = calculate_pixel_size(coeffs, args.distance)
        print(f"Pixel size at {args.distance} units:")
        print(f"  Width:  {pixel_w:.4f} units/pixel")
        print(f"  Height: {pixel_h:.4f} units/pixel")

        # Example measurements
        example_pixels = [10, 50, 100, 200]
        print(f"\nExample measurements at {args.distance} units distance:")
        for pixels in example_pixels:
            real_size = pixels_to_real_size(pixels, coeffs, args.distance, "horizontal")
            print(f"  {pixels:3d} pixels = {real_size:.2f} units")

    # Method 2: Reference object (will be set interactively if reference_real is provided)
    reference_pixels = None
    if args.reference_real is not None:
        print(f"\nMethod 2: Using reference object")
        print("-" * 60)
        print(f"Reference object real size: {args.reference_real} units")
        print("(You will click two points on the reference object to measure it)")
        reference_pixels = None  # Will be set interactively

    # Step 1: Interactive reference calibration (if needed)
    scale = None
    scale_method = "pixels only"

    if args.reference_real is not None and reference_pixels is None:
        print("\n" + "="*60)
        print("STEP 1: REFERENCE OBJECT CALIBRATION")
        print("="*60)
        print(f"\nClick TWO points on the reference object ({args.reference_real} units)")
        print("The points can be at any orientation (not just horizontal/vertical)")
        print("\nInstructions:")
        print("  - Click point 1 (start of reference object)")
        print("  - Click point 2 (end of reference object)")
        print("  - Press 'r' to reset and try again")
        print("  - Press 'q' to quit")

        ref_points = []
        ref_display = undistorted.copy()

        def ref_mouse_callback(event, x, y, flags, param):
            nonlocal ref_points, ref_display, reference_pixels

            if event == cv2.EVENT_LBUTTONDOWN:
                ref_points.append((x, y))

                # Draw point
                cv2.circle(ref_display, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(ref_display, f"P{len(ref_points)}", (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if len(ref_points) == 2:
                    # Draw line
                    cv2.line(ref_display, ref_points[0], ref_points[1], (0, 0, 255), 2)

                    # Calculate distance
                    pixel_dist = np.linalg.norm(np.array(ref_points[1]) - np.array(ref_points[0]))
                    reference_pixels = pixel_dist

                    # Display measurement
                    mid_x = (ref_points[0][0] + ref_points[1][0]) // 2
                    mid_y = (ref_points[0][1] + ref_points[1][1]) // 2

                    cv2.putText(ref_display, f"{pixel_dist:.1f}px = {args.reference_real}u",
                               (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 0, 255), 2)

                    print(f"\nReference object measured:")
                    print(f"  Point 1: {ref_points[0]}")
                    print(f"  Point 2: {ref_points[1]}")
                    print(f"  Distance: {pixel_dist:.1f} pixels = {args.reference_real} units")
                    print(f"\nPress any key to continue to measurement mode...")

        cv2.namedWindow("Reference Calibration")
        cv2.setMouseCallback("Reference Calibration", ref_mouse_callback)

        while reference_pixels is None:
            cv2.imshow("Reference Calibration", ref_display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cv2.destroyAllWindows()
                print("\nCancelled.")
                return
            elif key == ord('r'):
                # Reset
                ref_points = []
                ref_display = undistorted.copy()
                print("\nReset - click two points again")

        # Wait for user to acknowledge
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Calculate scale
        scale = calculate_scale_from_reference(reference_pixels, args.reference_real)
        scale_method = "reference object"

        print(f"\nScale calculated: {scale:.4f} units/pixel")

    # Interactive measurement mode
    print("\n" + "="*60)
    if args.reference_real is not None:
        print("STEP 2: MEASUREMENT MODE")
    else:
        print("INTERACTIVE MEASUREMENT MODE")
    print("="*60)
    print("\nInstructions:")
    print("  - Click two points to measure distance")
    print("  - Press 'r' to reset")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save image with measurements")

    # Determine scale (if not already set from reference calibration)
    if scale is None:
        if args.distance is not None:
            pixel_w, _ = calculate_pixel_size(coeffs, args.distance)
            scale = pixel_w
            scale_method = f"distance={args.distance}"
        else:
            print("\nWarning: No scale available. Measurements will be in pixels only.")
            print("Use --distance or --reference-real for real-world measurements.")
            scale_method = "pixels only"

    # Interactive measurement
    points = []
    display_img = undistorted.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, display_img

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

            # Draw point
            cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)

            # If we have two points, draw line and measure
            if len(points) == 2:
                cv2.line(display_img, points[0], points[1], (0, 255, 0), 2)

                # Calculate distance
                pixel_dist = np.linalg.norm(np.array(points[1]) - np.array(points[0]))

                # Display measurement
                mid_x = (points[0][0] + points[1][0]) // 2
                mid_y = (points[0][1] + points[1][1]) // 2

                if scale is not None:
                    real_dist = pixel_dist * scale
                    text = f"{pixel_dist:.1f}px = {real_dist:.2f}u"
                else:
                    text = f"{pixel_dist:.1f}px"

                cv2.putText(display_img, text, (mid_x, mid_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                print(f"\nMeasurement:")
                print(f"  Point 1: {points[0]}")
                print(f"  Point 2: {points[1]}")
                print(f"  Distance: {pixel_dist:.1f} pixels")
                if scale is not None:
                    print(f"  Real distance: {real_dist:.2f} units ({scale_method})")

                # Reset points for next measurement
                points = []

    cv2.namedWindow("Measurement Tool")
    cv2.setMouseCallback("Measurement Tool", mouse_callback)

    while True:
        cv2.imshow("Measurement Tool", display_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset
            points = []
            display_img = undistorted.copy()
            print("\nReset measurements")
        elif key == ord('s'):
            # Save
            output_path = args.image.replace('.', '_measured.')
            cv2.imwrite(output_path, display_img)
            print(f"\nSaved to: {output_path}")

    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == "__main__":
    main()
