"""Configuration for AI Meter Reading project."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class DataConfig:
    """Configuration for data paths and parameters."""

    base_dir: Path
    raw_images_dir: Path
    crops_dir: Path
    aligned_dir: Path
    final_crops_dir: Path
    labels_csv: Optional[Path] = None
    submission_csv: Optional[Path] = None
    final_size: tuple[int, int] = (128, 384)

    def __post_init__(self):
        """Create output directories if they don't exist."""
        for path in [self.crops_dir, self.aligned_dir, self.final_crops_dir]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class YOLOConfig:
    """Configuration for YOLO training."""

    yaml_path: Path
    train_images_dir: Path
    val_images_dir: Path
    class_names: list[str]
    weights_path: Path | None = None
    epochs: int = 30
    imgsz: int = 640
    batch: int = 8
    optimizer: str = "AdamW"
    lr0: float = 0.002
    momentum: float = 0.9
    conf_threshold: float = 0.3


@dataclass
class ModelConfig:
    """Configuration for model training."""

    device: str = "cuda"
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-3
    dropout: float = 0.3
    input_size: tuple[int, int] = (128, 384)


@dataclass
class OCRConfig:
    """Configuration for OCR."""

    languages: list[str] = None
    use_gpu: bool = True
    n_digits: int = 3

    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en"]


def load_config_from_dict(config_dict: Dict) -> tuple[DataConfig, YOLOConfig, YOLOConfig]:
    """Load configuration from dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        Tuple of (DataConfig, YOLOConfig_crop, YOLOConfig_align)
    """
    base_dir = Path(config_dict.get("base_dir", "."))

    data_config = DataConfig(
        base_dir=base_dir,
        raw_images_dir=base_dir / config_dict.get("raw_images_dir", "pictures"),
        crops_dir=base_dir / config_dict.get("crops_dir", "crops"),
        aligned_dir=base_dir / config_dict.get("aligned_dir", "aligned"),
        final_crops_dir=base_dir / config_dict.get("final_crops_dir", "final_crops"),
        labels_csv=Path(config_dict["labels_csv"]) if "labels_csv" in config_dict else None,
        submission_csv=Path(config_dict.get("submission_csv", base_dir / "submission.csv")),
    )

    yolo_crop_config = YOLOConfig(
        yaml_path=base_dir / config_dict.get("yolo_crop_yaml", "data_crop.yaml"),
        train_images_dir=base_dir / config_dict.get("yolo_crop_train_images", "cadrants.v4i.yolov8/train/images"),
        val_images_dir=base_dir / config_dict.get("yolo_crop_val_images", "cadrants.v4i.yolov8/val/images"),
        class_names=["chiffres"],
        weights_path=Path(config_dict.get("yolo_crop_weights")) if "yolo_crop_weights" in config_dict else None,
    )

    yolo_align_config = YOLOConfig(
        yaml_path=base_dir / config_dict.get("yolo_align_yaml", "data_align.yaml"),
        train_images_dir=base_dir / config_dict.get("yolo_align_train_images", "boites.v3i.yolov8/train/images"),
        val_images_dir=base_dir / config_dict.get("yolo_align_val_images", "boites.v3i.yolov8/val/images"),
        class_names=["noir", "rouge"],
        weights_path=Path(config_dict.get("yolo_align_weights")) if "yolo_align_weights" in config_dict else None,
    )

    return data_config, yolo_crop_config, yolo_align_config
