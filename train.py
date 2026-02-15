"""
Training script for the Vision Transformer on CIFAR-10.

Usage:
    python train.py                          # Train with default settings
    python train.py --preset tiny --epochs 5 # Quick experiment with tiny model
    python train.py --preset small --lr 1e-4 # Larger model with lower LR
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import build_vit
from dataset import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train Vision Transformer on CIFAR-10")
    parser.add_argument("--preset", type=str, default="default",
                        choices=["tiny", "default", "small"],
                        help="Model size preset")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="Weight decay for AdamW")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Dataset directory")
    parser.add_argument("--save-dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Data loader workers")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples:     {len(test_loader.dataset)}")

    # Model
    model = build_vit(preset=args.preset, num_classes=10).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model preset: {args.preset} | Parameters: {param_count:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0

    # Training loop
    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Test Loss':>9} | {'Test Acc':>8} | {'LR':>10} | {'Time':>6}")
    print("-" * 78)

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2f}% | "
              f"{test_loss:9.4f} | {test_acc:7.2f}% | {lr:10.6f} | {elapsed:5.1f}s")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc,
                "preset": args.preset,
            }, os.path.join(args.save_dir, "best_model.pt"))

    # Save final model
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_acc": test_acc,
        "preset": args.preset,
    }, os.path.join(args.save_dir, "final_model.pt"))

    print(f"\nTraining complete. Best test accuracy: {best_acc:.2f}%")
    print(f"Checkpoints saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
