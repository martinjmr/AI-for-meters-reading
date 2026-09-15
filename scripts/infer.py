"""Complete inference pipeline for meter reading."""

import argparse
import logging
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

from src.config import DataConfig, YOLOConfig
from src.ocr import MeterOCR, predict_folder
from src.preprocessing import crop_dial_zone, align_dial, build_final_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_inference_pipeline(
    raw_images_dir: str,
    crop_weights: str,
    align_weights: str,
    output_dir: str,
    save_intermediate: bool = False
) -> None:
    """Run complete inference pipeline.

    Args:
        raw_images_dir: Path to input images
        crop_weights: Path to YOLO crop model weights
        align_weights: Path to YOLO align model weights
        output_dir: Path to save results
        save_intermediate: Whether to save intermediate results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting meter reading inference pipeline")
    logger.info("=" * 60)

    # Step 1: Crop dial zones
    logger.info("\n[Step 1/4] Cropping dial zones...")
    crops_dir = output_path / "crops"
    try:
        crop_model = YOLO(crop_weights)
        n_crop, n_crop_miss = crop_dial_zone(crop_model, raw_images_dir, str(crops_dir))
    except Exception as e:
        logger.error(f"Cropping failed: {e}")
        return

    # Step 2: Align dials
    logger.info("\n[Step 2/4] Aligning dials...")
    aligned_dir = output_path / "aligned"
    try:
        align_model = YOLO(align_weights)
        n_align, n_align_miss = align_dial(align_model, str(crops_dir), str(aligned_dir))
    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        return

    # Step 3: Build final dataset
    logger.info("\n[Step 3/4] Building final dataset...")
    final_crops_dir = output_path / "final_crops"
    try:
        n_final = build_final_dataset(str(aligned_dir), str(final_crops_dir))
    except Exception as e:
        logger.error(f"Dataset building failed: {e}")
        return

    # Step 4: OCR recognition
    logger.info("\n[Step 4/4] Running OCR predictions...")
    try:
        use_gpu = torch.cuda.is_available()
        ocr = MeterOCR(use_gpu=use_gpu)
        predictions_df = predict_folder(str(final_crops_dir), ocr.reader)

        # Save predictions
        output_csv = output_path / "predictions.csv"
        predictions_df.to_csv(output_csv, index=False)
        logger.info(f"✅ Predictions saved to {output_csv}")

    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Input images: {n_crop + n_crop_miss}")
    logger.info(f"Cropped: {n_crop} ({100*n_crop/(n_crop+n_crop_miss):.1f}%)")
    logger.info(f"Aligned: {n_align} ({100*n_align/(n_align+n_align_miss):.1f}%)")
    logger.info(f"Predictions: {len(predictions_df)}")
    logger.info(f"Output directory: {output_path}")

    if not save_intermediate:
        logger.info("Cleaning up intermediate files...")
        import shutil
        shutil.rmtree(crops_dir, ignore_errors=True)
        shutil.rmtree(aligned_dir, ignore_errors=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run meter reading inference pipeline"
    )
    parser.add_argument(
        "raw_images_dir",
        help="Path to directory with input meter images"
    )
    parser.add_argument(
        "--crop-weights",
        required=True,
        help="Path to YOLO model for dial cropping"
    )
    parser.add_argument(
        "--align-weights",
        required=True,
        help="Path to YOLO model for dial alignment"
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory (default: results)"
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate processed images"
    )

    args = parser.parse_args()

    if not Path(args.raw_images_dir).exists():
        logger.error(f"Input directory not found: {args.raw_images_dir}")
        sys.exit(1)

    if not Path(args.crop_weights).exists():
        logger.error(f"Crop weights not found: {args.crop_weights}")
        sys.exit(1)

    if not Path(args.align_weights).exists():
        logger.error(f"Align weights not found: {args.align_weights}")
        sys.exit(1)

    run_inference_pipeline(
        args.raw_images_dir,
        args.crop_weights,
        args.align_weights,
        args.output,
        save_intermediate=args.keep_intermediate
    )


if __name__ == "__main__":
    main()
