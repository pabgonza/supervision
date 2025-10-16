"""
Image Undistortion Script

This script undistorts images using previously calculated camera calibration coefficients.
It loads the calibration data from a JSON file and applies lens distortion correction
to all images in a specified directory.

Usage:
    python undistort_images.py --images ./images_to_undistort --camera_coefs camera_calibration.json --output ./undistorted --crop
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from calibrate_camera import (
    CameraCoefficients,
    undistort_image
)

def main():
    parser = argparse.ArgumentParser(description="Images undistortion using camera calibration parameters")
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Directory containing images to be undistorted"
    )
    parser.add_argument(
        "--camera_coefs",
        type=str,
        default="camera_calibration.json",
        help="Input file for calibration coefficients (default: camera_calibration.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="undistorted_images",
        help="Output directory for undistorted images (default: undistorted_images)"
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Crop the undistorted images to remove black borders"
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=0.0,
        help="Balance/alpha parameter for undistortion (0.0=minimal black pixels, 1.0=retain all pixels)"
    )

    args = parser.parse_args()

    # Load camera calibration coefficients
    print(f"Loading calibration coefficients from: {args.camera_coefs}")
    try:
        coeffs = CameraCoefficients.load(args.camera_coefs)
        print(f"\n{coeffs}\n")
    except FileNotFoundError:
        print(f"Error: Calibration file not found: {args.camera_coefs}")
        return
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        return

    # Get input images
    input_dir = Path(args.images)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {args.images}")
        return

    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    if not image_files:
        print(f"No images found in {args.images}")
        return

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Crop black borders: {args.crop}")
    print(f"Balance parameter: {args.balance}")
    print(f"Found {len(image_files)} images to process\n")

    # Process each image
    successful = 0
    for img_path in image_files:
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [X] Failed to load: {img_path.name}")
                continue

            # Undistort image
            undistorted = undistort_image(img, coeffs, crop=args.crop, balance=args.balance)

            # Save undistorted image
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), undistorted)

            successful += 1
            print(f"  [{successful}] Processed: {img_path.name} -> {output_path.name}")

        except Exception as e:
            print(f"  [X] Error processing {img_path.name}: {e}")

    print(f"\nSuccessfully processed {successful}/{len(image_files)} images")
    print(f"Undistorted images saved to: {output_dir}")


if __name__ == "__main__":
    main()
