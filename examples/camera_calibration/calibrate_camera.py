"""
Camera Calibration Script

This script performs camera calibration using a chessboard pattern.
It calculates camera matrix and distortion coefficients that can be used
to correct lens distortion in images or video streams.

Usage:
    python calibrate_camera.py --images ./calibration_images --width 9 --height 6
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


@dataclass
class CameraCoefficients:
    """Camera calibration coefficients."""

    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Optical center x
    cy: float  # Optical center y
    k1: float  # Radial distortion coefficient 1
    k2: float  # Radial distortion coefficient 2
    p1: float  # Tangential distortion coefficient 1
    p2: float  # Tangential distortion coefficient 2
    k3: float  # Radial distortion coefficient 3
    width: int  # Image width
    height: int  # Image height
    error: float  # Re-projection error

    def __str__(self) -> str:
        return (
            f"Camera Matrix:\n"
            f"  fx: {self.fx:.2f}\n"
            f"  fy: {self.fy:.2f}\n"
            f"  cx: {self.cx:.2f}\n"
            f"  cy: {self.cy:.2f}\n"
            f"Distortion Coefficients:\n"
            f"  k1: {self.k1:.6f}\n"
            f"  k2: {self.k2:.6f}\n"
            f"  p1: {self.p1:.6f}\n"
            f"  p2: {self.p2:.6f}\n"
            f"  k3: {self.k3:.6f}\n"
            f"Image Size: {self.width}x{self.height}\n"
            f"Re-projection Error: {self.error:.4f}"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, path: str) -> None:
        """Save coefficients to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Calibration saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "CameraCoefficients":
        """Load coefficients from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


def extract_chessboard_corners(
    image_dir: str,
    chessboard_size: Tuple[int, int],
    square_size: float = 1.0,
    show_corners: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
    """
    Extract chessboard corners from calibration images.

    Args:
        image_dir: Directory containing calibration images
        chessboard_size: Number of inner corners (width, height)
        square_size: Size of chessboard square in your preferred unit (e.g., mm)
        show_corners: Whether to display detected corners

    Returns:
        Tuple of (object_points, image_points, image_size)
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (width-1,height-1,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    image_size = None

    # Get list of calibration images
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    if not image_files:
        raise ValueError(f"No images found in {image_dir}")

    print(f"Found {len(image_files)} calibration images")
    print(f"Looking for chessboard with {chessboard_size[0]}x{chessboard_size[1]} inner corners")

    successful_detections = 0

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            objpoints.append(objp)

            # Refine corner positions to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)

            successful_detections += 1
            print(f"  [{successful_detections}] Found corners in: {img_path.name}")

            # Draw and display corners if requested
            if show_corners:
                cv2.drawChessboardCorners(img, chessboard_size, corners_refined, ret)
                cv2.imshow('Chessboard Corners', img)
                cv2.waitKey(500)
        else:
            print(f"  [X] Failed to find corners in: {img_path.name}")

    if show_corners:
        cv2.destroyAllWindows()

    if successful_detections == 0:
        raise ValueError("No chessboard corners detected in any image")

    print(f"\nSuccessfully detected corners in {successful_detections}/{len(image_files)} images")

    return objpoints, imgpoints, image_size


def calibrate_camera(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    image_size: Tuple[int, int]
) -> CameraCoefficients:
    """
    Calibrate camera using detected chessboard corners.

    Args:
        objpoints: List of 3D object points
        imgpoints: List of 2D image points
        image_size: Size of images (width, height)

    Returns:
        CameraCoefficients object
    """
    print("\nCalibrating camera...")

    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # Calculate re-projection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    mean_error = total_error / len(objpoints)

    # Create CameraCoefficients object
    coeffs = CameraCoefficients(
        fx=camera_matrix[0, 0],
        fy=camera_matrix[1, 1],
        cx=camera_matrix[0, 2],
        cy=camera_matrix[1, 2],
        k1=dist_coeffs[0, 0],
        k2=dist_coeffs[0, 1],
        p1=dist_coeffs[0, 2],
        p2=dist_coeffs[0, 3],
        k3=dist_coeffs[0, 4],
        width=image_size[0],
        height=image_size[1],
        error=mean_error
    )

    print("\nCalibration complete!")
    print(f"\n{coeffs}")

    return coeffs


def undistort_image(
    image: np.ndarray,
    coeffs: CameraCoefficients,
    crop: bool = True
) -> np.ndarray:
    """
    Undistort an image using calibration coefficients.

    Args:
        image: Input image
        coeffs: Camera calibration coefficients
        crop: Whether to crop the result to remove black borders

    Returns:
        Undistorted image
    """
    h, w = image.shape[:2]

    # Build camera matrix and distortion coefficients
    camera_matrix = np.array([
        [coeffs.fx, 0, coeffs.cx],
        [0, coeffs.fy, coeffs.cy],
        [0, 0, 1]
    ])

    dist_coeffs = np.array([coeffs.k1, coeffs.k2, coeffs.p1, coeffs.p2, coeffs.k3])

    if crop:
        # Get optimal new camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )

        # Undistort
        dst = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # Crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
    else:
        # Undistort without cropping
        dst = cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)

    return dst


def main():
    parser = argparse.ArgumentParser(description="Camera calibration using chessboard pattern")
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Directory containing calibration images"
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Number of inner corners along the width of the chessboard"
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Number of inner corners along the height of the chessboard"
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=1.0,
        help="Size of chessboard square (default: 1.0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="camera_calibration.json",
        help="Output file for calibration coefficients (default: camera_calibration.json)"
    )
    parser.add_argument(
        "--show-corners",
        action="store_true",
        help="Display detected corners during processing"
    )
    parser.add_argument(
        "--test-image",
        type=str,
        default=None,
        help="Test calibration on a specific image"
    )

    args = parser.parse_args()

    # Extract chessboard corners
    objpoints, imgpoints, image_size = extract_chessboard_corners(
        args.images,
        (args.width, args.height),
        args.square_size,
        args.show_corners
    )

    # Calibrate camera
    coeffs = calibrate_camera(objpoints, imgpoints, image_size)

    # Save calibration
    coeffs.save(args.output)

    # Test calibration on an image if provided
    if args.test_image:
        print(f"\nTesting calibration on: {args.test_image}")
        img = cv2.imread(args.test_image)

        if img is None:
            print(f"Error: Could not load image {args.test_image}")
            return

        undistorted = undistort_image(img, coeffs, crop=True)

        # Display comparison
        h, w = img.shape[:2]
        undistorted_resized = cv2.resize(undistorted, (w, h))
        comparison = np.hstack([img, undistorted_resized])

        cv2.imshow('Original (Left) vs Undistorted (Right)', comparison)
        print("Press any key to close the comparison window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save undistorted image
        output_path = Path(args.test_image).stem + "_undistorted.jpg"
        cv2.imwrite(output_path, undistorted)
        print(f"Undistorted image saved to: {output_path}")


if __name__ == "__main__":
    main()
