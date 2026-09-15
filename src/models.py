"""Neural network models for digit recognition."""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger(__name__)


class SimpleMultiDigitCNN(nn.Module):
    """Simple CNN with 3 independent heads for digit recognition.

    Architecture:
    - 3 convolutional blocks with BatchNorm, ReLU, MaxPool
    - Shared fully connected layer
    - 3 independent output heads (one per digit)
    - 30% dropout regularization
    """

    def __init__(self, input_hw: Tuple[int, int] = (128, 384)):
        """Initialize CNN model.

        Args:
            input_hw: Input size as (height, width)
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        h, w = input_hw
        flat_dim = 128 * (h // 8) * (w // 8)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.fc_shared = nn.Linear(flat_dim, 512)

        # Three independent heads for each digit
        self.digit1 = nn.Linear(512, 10)
        self.digit2 = nn.Linear(512, 10)
        self.digit3 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Tuple of output tensors for each digit (batch_size, 10)
        """
        x = self.features(x)
        x = self.flatten(x)
        x = self.dropout(F.relu(self.fc_shared(x)))
        return self.digit1(x), self.digit2(x), self.digit3(x)


class ResNetMultiDigit(nn.Module):
    """ResNet-based model with 3 independent output heads.

    Uses a pre-trained ResNet backbone with frozen lower layers and
    3 independent heads for digit prediction.
    """

    def __init__(self, backbone: str = "resnet18", freeze_until: int = 9):
        """Initialize ResNet multi-digit model.

        Args:
            backbone: Name of ResNet variant ('resnet18', 'resnet34', etc.)
            freeze_until: Number of layers to freeze from start
        """
        super().__init__()

        base_model = getattr(models, backbone)(weights="DEFAULT")
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        in_features = base_model.fc.in_features

        # Three independent output heads
        self.fc1 = nn.Linear(in_features, 10)
        self.fc2 = nn.Linear(in_features, 10)
        self.fc3 = nn.Linear(in_features, 10)

        # Freeze lower layers
        for i, child in enumerate(self.features.children()):
            if i < freeze_until:
                for param in child.parameters():
                    param.requires_grad = False

        logger.info(f"ResNet initialized with {freeze_until} layers frozen")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Tuple of output tensors for each digit (batch_size, 10)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x), self.fc2(x), self.fc3(x)
