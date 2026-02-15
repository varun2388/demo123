"""
Inference script for the Vision Transformer.

Usage:
    python predict.py --image path/to/image.png
    python predict.py --image photo.jpg --checkpoint checkpoints/best_model.pt
    python predict.py --evaluate  # Run full test-set evaluation
"""

import argparse

import torch
from torchvision import transforms
from PIL import Image

from model import build_vit
from dataset import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, get_dataloaders


def load_model(checkpoint_path, device):
    """Load a trained ViT model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    preset = checkpoint.get("preset", "default")
    model = build_vit(preset=preset, num_classes=10).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model (preset={preset}) from {checkpoint_path}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Checkpoint test accuracy: {checkpoint.get('test_acc', 'N/A'):.2f}%")
    return model


def preprocess_image(image_path):
    """Load and preprocess a single image for inference."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict_image(model, image_path, device, top_k=5):
    """Predict the class of a single image."""
    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)

    top_probs, top_indices = probs.topk(top_k, dim=1)

    print(f"\nPredictions for: {image_path}")
    print("-" * 40)
    for i in range(top_k):
        idx = top_indices[0, i].item()
        prob = top_probs[0, i].item()
        print(f"  {CIFAR10_CLASSES[idx]:>12s}: {prob:6.2%}")

    return top_indices[0, 0].item()


@torch.no_grad()
def evaluate_test_set(model, device, data_dir="./data"):
    """Run evaluation on the full CIFAR-10 test set."""
    _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=128)

    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for pred, label in zip(preds, labels):
            class_total[label.item()] += 1
            if pred == label:
                class_correct[label.item()] += 1

    overall_acc = 100.0 * correct / total
    print(f"\nOverall test accuracy: {overall_acc:.2f}% ({correct}/{total})")
    print(f"\nPer-class accuracy:")
    print("-" * 35)
    for i, name in enumerate(CIFAR10_CLASSES):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            print(f"  {name:>12s}: {acc:6.2f}% ({class_correct[i]}/{class_total[i]})")

    return overall_acc


def main():
    parser = argparse.ArgumentParser(description="ViT Inference on CIFAR-10")
    parser.add_argument("--image", type=str, help="Path to an image to classify")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate on the full CIFAR-10 test set")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Dataset directory (for --evaluate)")
    args = parser.parse_args()

    if not args.image and not args.evaluate:
        parser.error("Provide --image or --evaluate (or both)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    if args.image:
        predict_image(model, args.image, device)

    if args.evaluate:
        evaluate_test_set(model, device, args.data_dir)


if __name__ == "__main__":
    main()
