"""Utility functions for image processing and geometry calculations."""

import logging
import math
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def compute_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Calculate center point of a bounding box.

    Args:
        box: Bounding box in format (x1, y1, x2, y2)

    Returns:
        Center point as (x, y) tuple
    """
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def compute_angle(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate angle of line from p1 to p2 relative to horizontal.

    Args:
        p1: First point (x, y)
        p2: Second point (x, y)

    Returns:
        Angle in degrees
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate an image around its center.

    Args:
        image: Input image as OpenCV array (BGR format)
        angle_deg: Rotation angle in degrees

    Returns:
        Rotated image as numpy array
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image, mat, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return rotated


def hsv_range_mask(image: np.ndarray,
                   sat_thresh: int = 90,
                   hue_low: Tuple[int, int] = (0, 10),
                   hue_high: Tuple[int, int] = (170, 180)) -> np.ndarray:
    """Create HSV mask for color detection.

    Args:
        image: Input image in BGR format
        sat_thresh: Saturation threshold
        hue_low: Low hue range (min, max)
        hue_high: High hue range (min, max)

    Returns:
        Binary mask
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask_low = (h >= hue_low[0]) & (h <= hue_low[1])
    mask_high = (h >= hue_high[0]) & (h <= hue_high[1])
    return (mask_low | mask_high) & (s >= sat_thresh)


def load_image_safe(image_path: str) -> np.ndarray | None:
    """Load image safely with error handling.

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array or None if failed
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            return None
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def save_image_safe(image_path: str, image: np.ndarray) -> bool:
    """Save image safely with error handling.

    Args:
        image_path: Path to save image
        image: Image as numpy array

    Returns:
        True if successful, False otherwise
    """
    try:
        cv2.imwrite(image_path, image)
        return True
    except Exception as e:
        logger.error(f"Error saving image to {image_path}: {e}")
        return False


def get_image_files(folder: str) -> list[str]:
    """Get list of image files from folder.

    Args:
        folder: Path to folder

    Returns:
        List of image filenames
    """
    import os
    return [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
