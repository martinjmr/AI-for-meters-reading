"""Training utilities for digit recognition models."""

import logging
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_digit_model(model: nn.Module,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     device: str,
                     epochs: int = 20,
                     lr: float = 1e-3) -> Tuple[nn.Module, list, list]:
    """Train a digit recognition model with 3 independent heads.

    Args:
        model: Model to train (should have 3 output heads)
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: Device to train on ('cuda' or 'cpu')
        epochs: Number of training epochs
        lr: Learning rate

    Returns:
        Tuple of (trained_model, train_losses, val_losses)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            out1, out2, out3 = model(imgs)
            loss = (
                criterion(out1, labels[:, 0]) +
                criterion(out2, labels[:, 1]) +
                criterion(out3, labels[:, 2])
            ) / 3

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                out1, out2, out3 = model(imgs)
                loss = (
                    criterion(out1, labels[:, 0]) +
                    criterion(out2, labels[:, 1]) +
                    criterion(out3, labels[:, 2])
                ) / 3
                running_val_loss += loss.item() * imgs.size(0)

        val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        logger.info(f"Epoch {epoch+1}/{epochs} — train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    return model, train_losses, val_losses


def evaluate_exact_accuracy(model: nn.Module,
                           loader: DataLoader,
                           device: str) -> float:
    """Evaluate model on exact accuracy (all 3 digits correct).

    Args:
        model: Model to evaluate
        loader: DataLoader with validation data
        device: Device to evaluate on

    Returns:
        Exact accuracy as float between 0 and 1
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluation"):
            imgs = imgs.to(device)
            out1, out2, out3 = model(imgs)
            preds = torch.stack([out1.argmax(1), out2.argmax(1), out3.argmax(1)], dim=1).cpu()
            correct += (preds == labels).all(dim=1).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    logger.info(f"✅ Exact accuracy (3 digits correct): {accuracy:.4f}")
    return accuracy


def plot_training_history(train_losses: list, val_losses: list,
                         save_path: Optional[str] = None) -> None:
    """Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.title("Training History", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")

    plt.show()
