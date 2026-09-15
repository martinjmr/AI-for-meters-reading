# Contributing to AI for Meter Reading

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR-USERNAME/AI-for-meters-reading.git
cd AI-for-meters-reading
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

## Development Workflow

### Making Changes

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Write or update tests for your changes
4. Ensure all tests pass: `make test`
5. Format and lint your code: `make format` and `make lint`

### Code Style Guidelines

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with:

- **Formatter:** [Black](https://github.com/psf/black) (line length: 100)
- **Linter:** [Flake8](https://github.com/PyCQA/flake8)
- **Import Sorter:** [isort](https://github.com/PyCQA/isort)

#### Example:

```python
from typing import Tuple

import numpy as np


def process_image(image: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, float]:
    """Process an image with given threshold.
    
    Args:
        image: Input image array
        threshold: Processing threshold
        
    Returns:
        Processed image and confidence score
    """
    result = image > threshold
    confidence = np.mean(result)
    return result, confidence
```

### Type Hints

All functions should include type hints:

```python
from typing import Optional, List, Dict

def get_predictions(
    image_path: str,
    model_path: Optional[str] = None
) -> Dict[str, float]:
    """Get model predictions for an image."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def train_model(
    train_data: np.ndarray,
    val_data: np.ndarray,
    epochs: int = 10
) -> None:
    """Train the model.
    
    Args:
        train_data: Training data array
        val_data: Validation data array
        epochs: Number of training epochs
        
    Raises:
        ValueError: If epochs < 1
    """
```

### Testing

Write tests for new functionality:

```python
import unittest
from src.utils import compute_angle

class TestComputeAngle(unittest.TestCase):
    def test_horizontal_line(self):
        angle = compute_angle((0, 0), (10, 0))
        self.assertAlmostEqual(angle, 0)
```

Run tests with: `make test`

## Commit Guidelines

Use clear, descriptive commit messages:

```
Good:
- "Add image preprocessing pipeline"
- "Fix OCR accuracy for rotated images"
- "Refactor YOLO model loading"

Avoid:
- "update" or "fix" (too vague)
- "asdf" or "temp" (meaningless)
```

## Pull Request Process

1. **Before submitting:**
   ```bash
   make format  # Format code
   make lint    # Check linting
   make test    # Run tests
   ```

2. **Create a clear PR title:** "Add feature: ..." or "Fix bug: ..."

3. **Fill the PR template:**
   - What changes were made?
   - Why are they needed?
   - How were they tested?
   - Any breaking changes?

4. **Wait for review** - maintainers will review and provide feedback

5. **Address feedback** - update your branch based on review comments

6. **Merge** - once approved, your PR will be merged!

## Reporting Issues

### Bug Reports

Include:
- Python version
- OS and environment
- Steps to reproduce
- Expected vs actual behavior
- Error messages/tracebacks
- Relevant code or screenshots

### Feature Requests

Describe:
- The use case
- Proposed solution
- Alternative approaches considered
- Why it's valuable to the project

## Documentation

- Update README.md for major changes
- Add docstrings to all new functions
- Update CHANGELOG if applicable
- Comment complex algorithms

## Questions?

- Check existing issues and PRs
- Ask in a new discussion/issue
- Reach out to maintainers

## Recognition

Contributors will be recognized in:
- This file's contributor list
- GitHub's contributor graph
- Project documentation

---

Thank you for making AI for Meter Reading better! 🚀
