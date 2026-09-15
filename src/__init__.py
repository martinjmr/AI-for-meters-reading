"""AI for Meter Reading - Automated water meter digit recognition."""

__version__ = "1.0.0"
__author__ = "Martin Jomier"

from src.preprocessing import (
    crop_dial_zone,
    align_dial,
    remove_red_zone,
    build_final_dataset,
    LetterboxResize,
)
from src.ocr import MeterOCR, read_digits, predict_folder
from src.models import SimpleMultiDigitCNN, ResNetMultiDigit

__all__ = [
    "crop_dial_zone",
    "align_dial",
    "remove_red_zone",
    "build_final_dataset",
    "LetterboxResize",
    "MeterOCR",
    "read_digits",
    "predict_folder",
    "SimpleMultiDigitCNN",
    "ResNetMultiDigit",
]
