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
from typing import List, Tuple, Optional, Union

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
    is_fisheye: bool = False  # Whether this is a fisheye lens calibration

    def __str__(self) -> str:
        lens_type = "Fisheye" if self.is_fisheye else "Standard"
        return (
            f"Lens Type: {lens_type}\n"
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
    image_size: Tuple[int, int],
    is_fisheye: bool = False
) -> CameraCoefficients:
    """
    Calibrate camera using detected chessboard corners.

    Args:
        objpoints: List of 3D object points
        imgpoints: List of 2D image points
        image_size: Size of images (width, height)
        is_fisheye: Whether to use fisheye calibration model (for FOV >= 180°)

    Returns:
        CameraCoefficients object
    """
    print(f"\nCalibrating camera ({'fisheye' if is_fisheye else 'standard'} model)...")

    if is_fisheye:
        # Fisheye calibration
        # Initialize camera matrix
        K = np.zeros((3, 3))
        K[0, 0] = image_size[0]  # fx
        K[1, 1] = image_size[1]  # fy
        K[0, 2] = image_size[0] / 2  # cx
        K[1, 2] = image_size[1] / 2  # cy
        K[2, 2] = 1.0

        # Initialize distortion coefficients
        D = np.zeros((4, 1))

        # Reshape object points and image points for fisheye
        # Fisheye expects shape (N, 1, 3) for objpoints and (N, 1, 2) for imgpoints
        objpoints_fisheye = [pts.reshape(-1, 1, 3) for pts in objpoints]
        imgpoints_fisheye = [pts.reshape(-1, 1, 2) for pts in imgpoints]

        # Calibration flags for fisheye
        calibration_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
            cv2.fisheye.CALIB_CHECK_COND +
            cv2.fisheye.CALIB_FIX_SKEW
        )

        # Criteria for calibration
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        # Calibrate fisheye camera
        ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints_fisheye, imgpoints_fisheye, image_size, K, D,
            flags=calibration_flags,
            criteria=criteria
        )

        # Calculate re-projection error
        total_error = 0
        for i in range(len(objpoints_fisheye)):
            imgpoints2, _ = cv2.fisheye.projectPoints(
                objpoints_fisheye[i], rvecs[i], tvecs[i], K, D
            )
            error = cv2.norm(imgpoints_fisheye[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error

        mean_error = total_error / len(objpoints_fisheye)

        # Create CameraCoefficients object
        # Fisheye distortion model uses k1, k2, k3, k4 (we map to k1, k2, p1, p2, k3)
        coeffs = CameraCoefficients(
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            k1=D[0, 0],
            k2=D[1, 0],
            p1=D[2, 0],
            p2=D[3, 0],
            k3=0.0,  # Fisheye uses 4 coefficients, we store in k1, k2, p1, p2
            width=image_size[0],
            height=image_size[1],
            error=mean_error,
            is_fisheye=True
        )
    else:
        # Standard pinhole calibration
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
            error=mean_error,
            is_fisheye=False
        )

    print("\nCalibration complete!")
    print(f"\n{coeffs}")

    return coeffs


def undistort_image(
    image: np.ndarray,
    coeffs: CameraCoefficients,
    crop: bool = True,
    balance: float = 0.0
) -> np.ndarray:
    """
    Undistort an image using calibration coefficients.

    Args:
        image: Input image
        coeffs: Camera calibration coefficients
        crop: Whether to crop the result to remove black borders (alpha=0 for standard, balance=0 for fisheye)
        balance: For fisheye only - balance parameter (0.0=no black pixels, 1.0=all source pixels retained)
                 For standard lenses, this is mapped to alpha parameter

    Returns:
        Undistorted image
    """
    h, w = image.shape[:2]

    # Build camera matrix
    camera_matrix = np.array([
        [coeffs.fx, 0, coeffs.cx],
        [0, coeffs.fy, coeffs.cy],
        [0, 0, 1]
    ])

    if coeffs.is_fisheye:
        # Fisheye undistortion
        # Build distortion coefficients (fisheye uses 4 coefficients)
        dist_coeffs = np.array([[coeffs.k1], [coeffs.k2], [coeffs.p1], [coeffs.p2]])

        # Determine balance value based on crop flag
        if crop:
            balance_value = balance  # Use provided balance (0.0 default = minimal black pixels)
        else:
            balance_value = 1.0  # Retain all source pixels

        # Scale camera matrix
        scaled_K = camera_matrix.copy()
        scaled_K[0, 0] = camera_matrix[0, 0] * w / coeffs.width
        scaled_K[1, 1] = camera_matrix[1, 1] * h / coeffs.height

        # Estimate new camera matrix for fisheye
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            scaled_K, dist_coeffs, (w, h), np.eye(3), balance=balance_value
        )

        # Create undistortion maps
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            scaled_K, dist_coeffs, np.eye(3), new_K, (w, h), cv2.CV_16SC2
        )

        # Apply undistortion using remap (more efficient)
        dst = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

    else:
        # Standard pinhole undistortion
        dist_coeffs = np.array([coeffs.k1, coeffs.k2, coeffs.p1, coeffs.p2, coeffs.k3])

        if crop:
            # alpha = 0: no black pixels, crops the image
            # Use balance parameter as alpha if provided, otherwise default to 0
            alpha = balance if balance > 0 else 0.0

            # Get optimal new camera matrix
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
            )

            # Undistort preserving original size
            dst = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

            ## Alternative Method: Using remap (more efficient for video streams)
            #map1, map2 = cv2.initUndistortRectifyMap(
            #    camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2
            #)
            #dst = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

        else:
            # alpha = 1: retain all source pixels with black borders
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 1.0, (w, h)
            )
            dst = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    return dst


def calculate_pixel_size(
    coeffs: CameraCoefficients,
    distance_mm: float
) -> Tuple[float, float]:
    """
    Calculate the size of one pixel in real-world units at a given distance.

    Args:
        coeffs: Camera calibration coefficients
        distance_mm: Distance from camera to object plane (in same units as square_size during calibration)

    Returns:
        Tuple of (pixel_width_mm, pixel_height_mm) - size of one pixel in real-world units

    Example:
        >>> coeffs = CameraCoefficients.load("camera_calibration.json")
        >>> pixel_w, pixel_h = calculate_pixel_size(coeffs, distance_mm=1000)
        >>> print(f"At 1000mm: 1 pixel = {pixel_w:.3f}mm × {pixel_h:.3f}mm")
    """
    # Formula: real_size = (pixel_size × distance) / focal_length
    # Therefore: pixel_size_in_mm = distance / focal_length
    pixel_width_mm = distance_mm / coeffs.fx
    pixel_height_mm = distance_mm / coeffs.fy

    return pixel_width_mm, pixel_height_mm


def pixels_to_real_size(
    pixel_distance: float,
    coeffs: CameraCoefficients,
    depth_mm: float,
    direction: str = "horizontal"
) -> float:
    """
    Convert a distance in pixels to real-world size at a given depth.

    Args:
        pixel_distance: Distance in pixels
        coeffs: Camera calibration coefficients
        depth_mm: Distance from camera to object (in same units as square_size during calibration)
        direction: "horizontal" (uses fx) or "vertical" (uses fy)

    Returns:
        Real-world size in same units as square_size

    Example:
        >>> coeffs = CameraCoefficients.load("camera_calibration.json")
        >>> # If object is 100 pixels wide and 500mm away
        >>> real_width = pixels_to_real_size(100, coeffs, 500, "horizontal")
        >>> print(f"Real width: {real_width:.2f}mm")
    """
    focal_length = coeffs.fx if direction == "horizontal" else coeffs.fy
    real_size = (pixel_distance * depth_mm) / focal_length
    return real_size


def calculate_scale_from_reference(
    reference_pixel_size: float,
    reference_real_size: float
) -> float:
    """
    Calculate scale factor using a reference object of known size.

    This is useful when you have an object of known size in the image at the same
    depth/plane as what you want to measure.

    Args:
        reference_pixel_size: Size of reference object in pixels
        reference_real_size: Known real size of reference object (in mm, cm, etc.)

    Returns:
        Scale factor (real_units_per_pixel)

    Example:
        >>> # You know a marker in the image is 50mm wide and measures 120 pixels
        >>> scale = calculate_scale_from_reference(120, 50)  # 0.4167 mm/pixel
        >>> # Now measure unknown object: 200 pixels × scale = 83.33mm
        >>> unknown_size = 200 * scale
        >>> print(f"Unknown object size: {unknown_size:.2f}mm")
    """
    return reference_real_size / reference_pixel_size


def estimate_distance_from_known_object(
    object_pixel_height: float,
    object_real_height: float,
    coeffs: CameraCoefficients
) -> float:
    """
    Estimate distance to an object of known size.

    Args:
        object_pixel_height: Height of object in pixels
        object_real_height: Known real height of object (in same units as square_size)
        coeffs: Camera calibration coefficients

    Returns:
        Estimated distance to object in same units as square_size

    Example:
        >>> coeffs = CameraCoefficients.load("camera_calibration.json")
        >>> # You see a person (1700mm tall) that appears as 400 pixels
        >>> distance = estimate_distance_from_known_object(400, 1700, coeffs)
        >>> print(f"Person is approximately {distance:.0f}mm away")
    """
    # Formula: distance = (real_height × focal_length) / pixel_height
    distance = (object_real_height * coeffs.fy) / object_pixel_height
    return distance


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
    parser.add_argument(
        "--fisheye",
        action="store_true",
        help="Use fisheye calibration model (for lenses with FOV >= 180°)"
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=0.0,
        help="Balance parameter for undistortion (0.0=minimal black pixels, 1.0=retain all pixels). Only used with --test-image"
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
    coeffs = calibrate_camera(objpoints, imgpoints, image_size, is_fisheye=args.fisheye)

    # Save calibration
    coeffs.save(args.output)

    # Test calibration on an image if provided
    if args.test_image:
        print(f"\nTesting calibration on: {args.test_image}")
        print(f"Using balance parameter: {args.balance}")
        img = cv2.imread(args.test_image)

        if img is None:
            print(f"Error: Could not load image {args.test_image}")
            return

        undistorted = undistort_image(img, coeffs, crop=True, balance=args.balance)

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
