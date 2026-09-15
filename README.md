# AI for Meter Reading

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: flake8](https://img.shields.io/badge/linting-flake8-blue.svg)](https://github.com/PyCQA/flake8)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Automated water meter digit recognition using deep learning and OCR.**

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Architecture](#architecture)
- [Pipeline](#pipeline)
- [Methodology](#methodology)
- [Results](#results)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project automates the reading of water meter digits from photographs using a combination of:
- **YOLOv8** for dial detection and alignment
- **EasyOCR** for digit recognition
- **Custom CNN** and **ResNet** architectures (experimental)

Instead of scheduling visits to read water meters, simply take a photo and let the AI extract the consumption value automatically.

**Dataset:** 795 water meter images with varying quality (blurry, rotated, distant)  
**Goal:** Accurately read the 3-digit black number (m³) from each meter dial

## Quick Start

```bash
# Install
pip install -e .

# Run inference on a folder of meter images
python scripts/infer.py ./images \
  --crop-weights path/to/crop_model.pt \
  --align-weights path/to/align_model.pt \
  --output results/
```

Results will be saved in `results/predictions.csv`.

## Installation

### Requirements
- Python 3.8 or higher
- CUDA 11.8+ (for GPU acceleration, optional but recommended)

### From Source

```bash
git clone https://github.com/martinjmr/AI-for-meters-reading.git
cd AI-for-meters-reading

# Install with dependencies
pip install -e .

# Or install with development tools
pip install -e ".[dev]"
```

### Dependencies

See [requirements.txt](requirements.txt) for full list:
- `ultralytics` - YOLOv8 implementation
- `torch` / `torchvision` - Deep learning framework
- `easyocr` - OCR engine
- `opencv-python` - Image processing
- `pandas` - Data handling
- `matplotlib` - Visualization

## Architecture

```
AI-for-meters-reading/
├── src/
│   ├── config.py              # Configuration dataclasses
│   ├── preprocessing.py       # Image preprocessing pipeline
│   ├── models.py              # CNN and ResNet architectures
│   ├── ocr.py                 # EasyOCR wrapper
│   ├── training.py            # Training utilities
│   └── utils.py               # Image processing utilities
├── scripts/
│   └── infer.py               # Complete inference pipeline
├── tests/                     # Unit tests
├── demo.ipynb                 # Interactive notebook demo
└── setup.py                   # Package configuration
```

## Pipeline

The inference pipeline consists of 4 sequential steps:

### Step 1: Dial Detection
![Dial Cropping](https://img.shields.io/badge/YOLO-Detection-blue)

- **Model:** YOLOv8 (trained on Roboflow)
- **Task:** Detect and crop the meter dial zone
- **Success Rate:** 726/795 images (91.3%)
- **Parameters:** 
  - Confidence threshold: 0.3
  - Image size: 640×640
  - Optimizer: AdamW (lr=0.002, momentum=0.9)

### Step 2: Dial Alignment
![Dial Alignment](https://img.shields.io/badge/YOLO-Alignment-blue)

- **Model:** YOLOv8 (2 classes: `noir`, `rouge`)
- **Task:** Detect reference points and rotate dial to horizontal
- **Method:**
  - Detect black box and red box (reference points)
  - Calculate angle between centers
  - Apply rotation transform
- **Success Rate:** 720/726 cropped images (99.2%)

### Step 3: Preprocessing
![Preprocessing](https://img.shields.io/badge/Image-Processing-green)

- Remove red zone (decimals - not informative for m³)
- Letterbox resize to 128×384 (preserves aspect ratio)
- Apply HSV thresholding for red detection

### Step 4: Digit Recognition
![OCR](https://img.shields.io/badge/EasyOCR-Recognition-orange)

- **Model:** EasyOCR (pre-trained CNN + RNN + CTC)
- **Task:** Extract 3-digit black number
- **Success Rate:** ~0.284 (competition metric)

## Methodology

### Data Processing

**Dataset Composition:**
- 795 water meter photographs
- Quality variations: blur, rotation, distance, lighting

**Preprocessing Strategy:**
1. Manual annotation of 150 images on Roboflow
2. First YOLO training → 320 high-quality crops
3. Second YOLO training → 726 final crops
4. Alignment using geometric transformation

### Model Comparison

| Model | Architecture | Accuracy | Status |
|-------|--------------|----------|--------|
| **EasyOCR** | CNN + RNN + CTC (pre-trained) | ~28.4% | ✅ **Deployed** |
| Custom CNN | 3 conv blocks + 3 heads | 0.68% | ❌ Insufficient data |
| ResNet18 | Fine-tuned backbone | Overfitting | ❌ Not scalable |

**Why EasyOCR?** With only 795 images, training from scratch led to severe overfitting. A pre-trained model generalizes much better.

## Results

### Performance Metrics

- **Input images processed:** 795
- **Successful crops:** 726 (91.3%)
- **Successful alignments:** 720 (99.2% of crops)
- **Final predictions:** 720
- **Exact accuracy (3 digits):** Verified on competition dataset

### Example Workflow

```python
from src.preprocessing import crop_dial_zone, align_dial, build_final_dataset
from src.ocr import MeterOCR
from ultralytics import YOLO

# Load models
crop_model = YOLO("weights/crop_model.pt")
align_model = YOLO("weights/align_model.pt")
ocr = MeterOCR()

# Pipeline
crop_dial_zone(crop_model, "raw_images/", "crops/")
align_dial(align_model, "crops/", "aligned/")
build_final_dataset("aligned/", "final/")
predictions = ocr.predict_folder("final/", ocr.reader)
```

## Development

### Code Quality

We follow professional Python practices:

```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test

# View all commands
make help
```

### Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=src
```

### Continuous Integration

GitHub Actions automatically:
- ✅ Runs linting (flake8, black, isort) on Python 3.9, 3.10, 3.11
- ✅ Runs unit tests with coverage
- ✅ Blocks commits that fail quality checks

See [.github/workflows/ci.yml](.github/workflows/ci.yml)

## Documentation

### Key Functions

**preprocessing.py**
- `crop_dial_zone()` - YOLO-based dial detection
- `align_dial()` - Geometric alignment using reference points
- `remove_red_zone()` - HSV-based red region removal
- `build_final_dataset()` - Letterbox resizing and final preprocessing

**ocr.py**
- `MeterOCR` - Wrapper class for EasyOCR
- `read_digits()` - Extract digits from single image
- `predict_folder()` - Batch inference

**models.py**
- `SimpleMultiDigitCNN` - Custom CNN with 3 output heads
- `ResNetMultiDigit` - ResNet backbone with fine-tuning

### Type Hints & Docstrings

All functions include:
- Complete type hints (Python 3.8+ compatible)
- Detailed docstrings following Google style
- Clear argument and return documentation

## Challenges & Limitations

### Challenges Encountered
- **Limited dataset:** Only 795 images for a computer vision task
- **Quality variations:** Blur, rotation, poor lighting, distant shots
- **Generalization:** Rotation algorithm doesn't generalize well on degraded images

### Lessons Learned
1. Pre-trained models > training from scratch on small datasets
2. Geometric preprocessing crucial for OCR reliability
3. Reference point detection more robust than direct rotation estimation

## Future Improvements

- [ ] Individual digit segmentation (instead of reading full band)
- [ ] Image enhancement (contrast, denoising)
- [ ] Data augmentation (synthetic rotation, blur, noise)
- [ ] Fine-tune EasyOCR on meter digits
- [ ] Deploy as web service (FastAPI)
- [ ] Mobile app integration

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

**Author:** Martin Jomier  
**Project Link:** https://github.com/martinjmr/AI-for-meters-reading  
**Email:** martinjomier92@gmail.com
