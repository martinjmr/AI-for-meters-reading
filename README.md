# AI for Meter Reading

With my team, I built a pipeline that reads the consumption (m³) of water meters from field photos: dial detection with YOLOv8, geometric alignment, then OCR. Computer-vision challenge of the IASO bachelor's programme (Université Paris Dauphine-PSL, June 2025), on Challenge Data.

| Stage | Method | Result |
|---|---|---|
| 1. Dial detection | YOLOv8 trained on 150 photos annotated in Roboflow, then retrained on 320 automatically cropped dials | 760 / 793 photos cropped (95.8%) |
| 2. Alignment | YOLOv8 locating the black and red reference boxes, then rotation | 585 dials aligned (77% of the crops) |
| 3. Digit reading | EasyOCR (pre-trained) on the black integer digits, keeping the last three | 28.4% of meters read exactly (public test set) |

The counts come from the output folders of the project. In a random sample of 60 crops, all 60 show the dial. Detection works on almost every photo, alignment loses about a quarter of the dials, and reading the digits is the bottleneck.

## Challenge leaderboard

The challenge is SUEZ's [AI for Meter Reading](https://challengedata.ens.fr/challenges/30) on Challenge Data. A reading counts as correct when its last three cubic-meter digits are exact; the score is the share of correct readings. The test set is split into a public part and a private part, which gives the final ranking.

| Test set | Our score | Organisers' benchmark | Rank |
|---|---|---|---|
| Public | 28.37% | 22.60% | 15th of 35 |
| Private (final) | 21.15% | 16.35% | 15th of 35 |

## Why a pre-trained OCR

| Model | Result |
|---|---|
| EasyOCR, pre-trained | 28.4% of meters read exactly (public test set) |
| Custom CNN (3 convolutional blocks, one head per digit) | Exact 3-digit accuracy 0.68% |
| ResNet18, fine-tuned | Overfitted, not kept |

With 793 photos, the networks we trained overfit, the small CNN trained from scratch as well as the fine-tuned ResNet18; a pre-trained OCR generalises much better.

## Pipeline

1. **Dial detection.** We annotated 150 photos by hand in Roboflow to train a first YOLOv8 model (one class), then used its 320 cleanest crops to train a second, more robust model. Settings: AdamW, learning rate 0.002, 640×640 images, confidence threshold 0.3.
2. **Alignment.** A second YOLOv8 model (classes `noir` and `rouge`) finds the black and red boxes; the angle between their centres gives the rotation that brings the digits horizontal.
3. **Preprocessing.** We remove the red zone (decimals, irrelevant for m³) by HSV thresholding, and letterbox the band to 128×384.
4. **Reading.** EasyOCR reads the black (integer) digits, and the last three are kept, since the challenge scores the last three cubic-meter digits.

## Limits and next steps

- The 793 photos are often blurred, rotated or taken from far away. The alignment step needs both reference boxes and fails on 175 of the 760 crops.
- Next steps: segment and read the digits one by one, augment the data (rotation, blur, noise), fine-tune the OCR on meter digits.

## Repository

- `AI_for_meter_reading_clean.ipynb`: the pipeline step by step (in French), reconstructed after the project from the team's working notebooks and not rerun end to end
- `src/`: the same steps as functions (preprocessing, OCR wrapper, CNN baselines)
- `scripts/infer.py`: runs the pipeline on a folder of photos
- `images/`: ten sample photos at three stages (raw photo, cropped dial, last three digits)

The challenge dataset and the trained YOLO weights are not included. With your own weights, from the repository root:

```bash
pip install -r requirements.txt
python -m scripts.infer path/to/photos --crop-weights crop.pt --align-weights align.pt --output results/
```
