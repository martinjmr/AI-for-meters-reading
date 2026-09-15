"""OCR functionality for digit recognition."""

import logging
import os
import re
from typing import List, Optional

import cv2
import easyocr
import pandas as pd
import torch
from tqdm import tqdm

from src.utils import get_image_files, load_image_safe

logger = logging.getLogger(__name__)


class MeterOCR:
    """Wrapper for EasyOCR with meter-specific configuration."""

    def __init__(self, languages: List[str] | None = None, use_gpu: bool = True):
        """Initialize OCR reader.

        Args:
            languages: List of language codes (default: ['en'])
            use_gpu: Whether to use GPU acceleration
        """
        if languages is None:
            languages = ["en"]

        self.languages = languages
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.reader = easyocr.Reader(languages, gpu=self.use_gpu)

        logger.info(f"EasyOCR initialized with languages: {languages}, GPU: {self.use_gpu}")

    def read_digits(self, image_path: str, n_digits: int = 3) -> str:
        """Read digits from image.

        Args:
            image_path: Path to image file
            n_digits: Expected number of digits

        Returns:
            String of detected digits (zero-padded if necessary)
        """
        img = load_image_safe(image_path)
        if img is None:
            return "ERR"

        try:
            result = self.reader.readtext(img, detail=0)
            raw = result[0] if result else ""

            # Extract only digits
            digits_only = re.sub(r"\D", "", raw)

            # Truncate or pad with zeros
            if len(digits_only) > n_digits:
                digits_only = digits_only[:n_digits]
            else:
                digits_only = digits_only.zfill(n_digits)

            return digits_only
        except Exception as e:
            logger.error(f"Error reading digits from {image_path}: {e}")
            return "ERR"


def read_digits(image_path: str, reader: easyocr.Reader, n_digits: int = 3) -> str:
    """Read digits from single image (functional interface).

    Args:
        image_path: Path to image
        reader: EasyOCR reader instance
        n_digits: Expected number of digits

    Returns:
        String of recognized digits
    """
    img = load_image_safe(image_path)
    if img is None:
        return "ERR"

    try:
        result = reader.readtext(img, detail=0)
        raw = result[0] if result else ""
        digits_only = re.sub(r"\D", "", raw)
        digits_only = digits_only[:n_digits] if len(digits_only) > n_digits else digits_only
        return digits_only.zfill(n_digits)
    except Exception as e:
        logger.error(f"Error reading {image_path}: {e}")
        return "ERR"


def predict_folder(image_folder: str, reader: easyocr.Reader,
                  n_digits: int = 3) -> pd.DataFrame:
    """Predict digits for all images in folder.

    Args:
        image_folder: Path to folder containing images
        reader: EasyOCR reader instance
        n_digits: Expected number of digits per image

    Returns:
        DataFrame with columns: ID, prediction
    """
    image_files = sorted(get_image_files(image_folder))
    rows = []

    for fname in tqdm(image_files, desc="OCR predictions"):
        pred = read_digits(os.path.join(image_folder, fname), reader, n_digits)
        rows.append({"ID": fname, "prediction": pred})

    df = pd.DataFrame(rows)
    logger.info(f"Predicted {len(df)} images")
    return df


def evaluate_predictions(predictions_df: pd.DataFrame,
                        labels_csv: str) -> float:
    """Evaluate predictions against ground truth labels.

    Args:
        predictions_df: DataFrame with 'ID' and 'prediction' columns
        labels_csv: Path to CSV file with ground truth labels

    Returns:
        Accuracy as float between 0 and 1
    """
    if not os.path.exists(labels_csv):
        logger.warning(f"Labels file not found: {labels_csv}")
        return 0.0

    labels_df = pd.read_csv(labels_csv)
    labels_df["target"] = labels_df["index_formate"].astype(str).str.zfill(3)

    merged = predictions_df.merge(labels_df[["ID", "target"]], on="ID", how="inner")
    merged["is_correct"] = merged["prediction"] == merged["target"]
    accuracy = merged["is_correct"].mean()

    logger.info(f"Exact accuracy (3 digits): {accuracy:.4f}")
    logger.info(f"Incorrect predictions: {(~merged['is_correct']).sum()}")

    return accuracy
