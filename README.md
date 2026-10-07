# AI for Meter Reading

Reading the consumption (m³) of water meters from field photos: dial detection with YOLOv8, geometric alignment, then OCR. Team project for a computer-vision challenge in the L3 IASO program (Université Paris Dauphine-PSL), **ranked 2nd in the cohort**.

| Stage | Method | Result |
|---|---|---|
| 1. Dial detection | YOLOv8 trained on 150 photos annotated in Roboflow, then retrained on 320 automatically cropped dials | 726 / 795 photos cropped (91.3%) |
| 2. Alignment | YOLOv8 locating the black and red reference boxes, then rotation | 720 / 726 dials aligned (99.2%) |
| 3. Digit reading | EasyOCR (pre-trained) on the 3-digit black band | Submission score ≈ 0.284 |

Detection and alignment work; reading the digits is the bottleneck.

## Why a pre-trained OCR

| Model | Result |
|---|---|
| EasyOCR, pre-trained | Submission score ≈ 0.284 |
| Custom CNN (3 convolutional blocks, one head per digit) | Exact 3-digit accuracy 0.68% |
| ResNet18, fine-tuned | Overfitted, not kept |

With 795 photos, networks trained from scratch overfit; a pre-trained OCR generalises much better.

## Pipeline

1. **Dial detection.** 150 photos annotated by hand in Roboflow trained a first YOLOv8 model (one class). Its 320 cleanest crops trained a second, more robust model. Settings: AdamW, learning rate 0.002, 640×640 images, confidence threshold 0.3.
2. **Alignment.** A second YOLOv8 model (classes `noir` and `rouge`) finds the black and red boxes; the angle between their centres gives the rotation that brings the digits horizontal.
3. **Preprocessing.** The red zone (decimals, irrelevant for m³) is removed by HSV thresholding, and the band is letterboxed to 128×384.
4. **Reading.** EasyOCR reads the 3-digit black number.

## Limits and next steps

- The 795 photos are often blurred, rotated or taken from far away, and the rotation step degrades on the worst ones.
- Next steps: segment and read the digits one by one, augment the data (rotation, blur, noise), fine-tune the OCR on meter digits.

## Repository

- `AI_for_meter_reading_clean.ipynb`: the full pipeline, step by step (in French)
- `src/`: the same steps as functions (preprocessing, OCR wrapper, CNN baselines)
- `scripts/infer.py`: runs the pipeline on a folder of photos
- `images/`: ten sample photos at three stages (raw, cropped dial, final band)

The challenge dataset and the trained YOLO weights are not included. With your own weights, from the repository root:

```bash
pip install -r requirements.txt
python -m scripts.infer path/to/photos --crop-weights crop.pt --align-weights align.pt --output results/
```

## Team

Built as a team for the L3 IASO computer-vision challenge at Université Paris Dauphine-PSL.
