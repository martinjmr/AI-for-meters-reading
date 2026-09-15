"""Setup configuration for AI Meter Reading package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai-meter-reading",
    version="1.0.0",
    author="Martin Jomier",
    author_email="martinjomier92@gmail.com",
    description="Automated water meter digit recognition using deep learning and OCR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/martinjmr/AI-for-meters-reading",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Graphics :: Viewers",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "easyocr>=1.7.0",
        "pandas>=1.5.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pillow>=9.5.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
    },
)
