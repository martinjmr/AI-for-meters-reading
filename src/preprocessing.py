"""Image preprocessing functions for meter reading."""

import logging
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from src.utils import (
    compute_angle,
    compute_center,
    get_image_files,
    hsv_range_mask,
    load_image_safe,
    rotate_image,
    save_image_safe,
)

logger = logging.getLogger(__name__)


def write_yolo_yaml(yaml_path: str, train_images_dir: str,
                   val_images_dir: str, class_names: list[str]) -> str:
    """Generate YOLO data.yaml configuration file.

    Args:
        yaml_path: Path to save YAML file
        train_images_dir: Path to training images
        val_images_dir: Path to validation images
        class_names: List of class names

    Returns:
        Path to created YAML file
    """
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    data_config = {
        "train": train_images_dir,
        "val": val_images_dir,
        "nc": len(class_names),
        "names": class_names,
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    logger.info(f"YAML configuration written: {yaml_path}")
    return yaml_path


def crop_dial_zone(model: YOLO, image_folder: str, output_folder: str,
                  conf: float = 0.3) -> Tuple[int, int]:
    """Detect and crop dial zones from images using YOLO.

    Keeps the highest confidence detection per image and crops to that region.

    Args:
        model: Trained YOLO model
        image_folder: Path to input images
        output_folder: Path to save cropped images
        conf: Confidence threshold for detection

    Returns:
        Tuple of (successful_crops, missed_detections)
    """
    os.makedirs(output_folder, exist_ok=True)
    image_files = get_image_files(image_folder)

    n_ok, n_missed = 0, 0

    for img_name in tqdm(image_files, desc="Cropping dial zones"):
        img_path = os.path.join(image_folder, img_name)
        img = Image.open(img_path).convert("RGB")

        results = model.predict(img_path, conf=conf, verbose=False)
        boxes = results[0].boxes

        if boxes is None or boxes.xyxy.shape[0] == 0:
            n_missed += 1
            logger.debug(f"No detection for {img_name}")
            continue

        best_idx = boxes.conf.argmax().item()
        x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx])
        cropped = img.crop((x1, y1, x2, y2))
        cropped.save(os.path.join(output_folder, img_name))
        n_ok += 1

    logger.info(f"✅ {n_ok} images cropped, {n_missed} missed (from {len(image_files)} total)")
    return n_ok, n_missed


def align_dial(model: YOLO, image_folder: str, output_folder: str,
              conf: float = 0.4) -> Tuple[int, int]:
    """Align dials using detected reference points (black/red boxes).

    Detects two reference points (black and red boxes), calculates rotation angle,
    and applies rotation to align the dial horizontally.

    Args:
        model: YOLO model trained to detect black and red reference boxes
        image_folder: Path to cropped images
        output_folder: Path to save aligned images
        conf: Confidence threshold for detection

    Returns:
        Tuple of (successful_alignments, missed_alignments)
    """
    os.makedirs(output_folder, exist_ok=True)
    image_files = get_image_files(image_folder)

    n_ok, n_missed = 0, 0
    CLASS_TO_ID = {"noir": 0, "rouge": 1}

    for fname in tqdm(image_files, desc="Aligning dials"):
        image_path = os.path.join(image_folder, fname)
        image = load_image_safe(image_path)
        if image is None:
            n_missed += 1
            continue

        results = model.predict(image_path, conf=conf, verbose=False)
        boxes = results[0].boxes

        best_boxes = {0: None, 1: None}  # 0 = noir, 1 = rouge
        for box in boxes:
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            if cls_id in best_boxes and (
                best_boxes[cls_id] is None or conf_score > best_boxes[cls_id][1]
            ):
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                best_boxes[cls_id] = ((x1, y1, x2, y2), conf_score)

        if best_boxes[0] is None or best_boxes[1] is None:
            n_missed += 1
            logger.debug(f"Missing reference point in {fname}")
            continue

        noir_center = compute_center(best_boxes[0][0])
        rouge_center = compute_center(best_boxes[1][0])
        angle = compute_angle(noir_center, rouge_center)

        rotated = rotate_image(image, angle)
        save_image_safe(os.path.join(output_folder, fname), rotated)
        n_ok += 1

    logger.info(f"✅ {n_ok} images aligned, {n_missed} skipped (from {len(image_files)} total)")
    return n_ok, n_missed


def remove_red_zone(image: np.ndarray, sat_thresh: int = 90,
                   hue_low: Tuple[int, int] = (0, 10),
                   hue_high: Tuple[int, int] = (170, 180)) -> np.ndarray:
    """Remove red zone (decimals) from meter dial.

    Detects red pixels using HSV thresholding and replaces them with black.
    The red zone represents decimal values which are not informative for m³ counting.

    Args:
        image: Input image in BGR format
        sat_thresh: Saturation threshold for red detection
        hue_low: Low hue range for red (in HSV)
        hue_high: High hue range for red (in HSV)

    Returns:
        Image with red zone removed (set to black)
    """
    red_mask = hsv_range_mask(image, sat_thresh, hue_low, hue_high)
    result = image.copy()
    result[red_mask] = (0, 0, 0)
    return result


class LetterboxResize:
    """Resize image preserving aspect ratio with letterboxing (black padding)."""

    def __init__(self, target_size: Tuple[int, int] = (128, 384), fill: int = 0):
        """Initialize letterbox resize transformer.

        Args:
            target_size: Target size as (height, width)
            fill: Fill value for padding (0 = black)
        """
        self.target_h, self.target_w = target_size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply letterbox resize to PIL image.

        Args:
            img: PIL Image to resize

        Returns:
            Resized PIL Image with padding
        """
        orig_w, orig_h = img.size
        scale = min(self.target_w / orig_w, self.target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img_resized = img.resize((new_w, new_h))

        pad_w = self.target_w - new_w
        pad_h = self.target_h - new_h
        padded = Image.new("RGB", (self.target_w, self.target_h), (self.fill,) * 3)
        padded.paste(img_resized, (pad_w // 2, pad_h // 2))
        return padded


def build_final_dataset(image_folder: str, output_folder: str,
                       target_size: Tuple[int, int] = (128, 384)) -> int:
    """Build final dataset with red zone removal and letterbox resizing.

    Args:
        image_folder: Path to aligned images
        output_folder: Path to save processed images
        target_size: Target size as (height, width)

    Returns:
        Number of images processed
    """
    os.makedirs(output_folder, exist_ok=True)
    letterbox = LetterboxResize(target_size)
    image_files = get_image_files(image_folder)

    for img_name in tqdm(image_files, desc="Building final dataset"):
        img_path = os.path.join(image_folder, img_name)
        cv_img = load_image_safe(img_path)
        if cv_img is None:
            continue

        cv_img = remove_red_zone(cv_img)
        pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        final_img = letterbox(pil_img)
        final_img.save(os.path.join(output_folder, img_name))

    logger.info(f"✅ Final dataset built: {len(image_files)} images in {output_folder}")
    return len(image_files)
