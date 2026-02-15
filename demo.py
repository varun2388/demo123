"""Quick demo of the Vision Transformer model."""

import torch
from model import build_vit
from dataset import CIFAR10_CLASSES

print("=" * 60)
print("  Vision Transformer (ViT) — Demo")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# --- 1. Build models and show architecture ---
print("\n--- Model Presets ---\n")
for preset in ["tiny", "default", "small"]:
    model = build_vit(preset=preset, num_classes=10)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {preset:>8s}: {params:>10,} parameters")

# --- 2. Load the default model and show structure ---
model = build_vit(preset="default", num_classes=10).to(device)
print(f"\nUsing 'default' preset for demo.")
print(f"\nModel architecture:")
print(model)

# --- 3. Forward pass with random input ---
print("\n--- Forward Pass (random 32x32 RGB image) ---\n")
dummy_input = torch.randn(1, 3, 32, 32).to(device)
with torch.no_grad():
    logits = model(dummy_input)
    probs = torch.softmax(logits, dim=1)

print(f"  Input shape:  {list(dummy_input.shape)}  (batch, channels, height, width)")
print(f"  Output shape: {list(logits.shape)}  (batch, num_classes)")
print(f"\n  Class probabilities (untrained, should be ~uniform):")
for i, name in enumerate(CIFAR10_CLASSES):
    print(f"    {name:>12s}: {probs[0, i].item():.2%}")

# --- 4. Batch inference on synthetic images ---
print("\n--- Batch Inference (8 synthetic images) ---\n")
batch_size = 8
fake_images = torch.randn(batch_size, 3, 32, 32).to(device)
fake_labels = torch.randint(0, 10, (batch_size,)).to(device)

with torch.no_grad():
    logits = model(fake_images)
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)

print(f"  {'Image':>7s}  {'True Label':>12s}  {'Predicted':>12s}  {'Confidence':>10s}")
print(f"  {'-'*47}")
for i in range(batch_size):
    true_label = CIFAR10_CLASSES[fake_labels[i].item()]
    pred_label = CIFAR10_CLASSES[preds[i].item()]
    conf = probs[i, preds[i]].item()
    print(f"  {i+1:>5d}    {true_label:>12s}  {pred_label:>12s}  {conf:>9.1%}")

print(f"\n  (Model is untrained — predictions are random as expected)")

# --- 5. Quick training loop ---
print("\n--- Quick Training Loop (10 batches of synthetic data) ---\n")
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader

# Create synthetic training data
syn_images = torch.randn(320, 3, 32, 32)
syn_labels = torch.randint(0, 10, (320,))
syn_loader = DataLoader(TensorDataset(syn_images, syn_labels), batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

model.train()
for i, (images, labels) in enumerate(syn_loader):
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    acc = (logits.argmax(1) == labels).float().mean().item()
    print(f"  Batch {i+1:>2d}/10  |  Loss: {loss.item():.4f}  |  Accuracy: {acc:.1%}")

print("\n  Training pipeline works end-to-end!")

print("\n" + "=" * 60)
print("  Demo complete!")
print("  To train fully: python train.py --preset default --epochs 50")
print("=" * 60)
