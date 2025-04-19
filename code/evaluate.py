"""
Evaluate a trained MiniXception model on a held‑out test set.

Supports plain evaluation **and** Test‑Time Augmentation (TTA): the
same image is fed through the network several times with random
horizontal flips; predictions are averaged to yield a more robust
score.
"""

import sys
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from model import MiniXception

TEST_PATH  = '../data/test'
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else '../results/best_model.pt'

# ------------------------------------------------------------------
# Dataset + transforms
# ------------------------------------------------------------------
test_transform = T.Compose([
    T.Grayscale(),
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

test_dataset = torchvision.datasets.ImageFolder(root=TEST_PATH,
                                                transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False,
                          num_workers=4, pin_memory=True)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def tta_predict(model, img, device, n_augment=3):
    """
    Run *n_augment* stochastic augmentations (currently only flips)
    on a single image and return the mean softmax probability tensor.
    """
    base  = img[0].cpu()       # shape: [C, H, W]
    preds = []

    for _ in range(n_augment):
        tta_tensor = torch.flip(base, dims=[2]) if torch.rand(1).item() < 0.5 else base
        tta_tensor = tta_tensor.unsqueeze(0).to(device)

        with autocast():
            out = model(tta_tensor)
        preds.append(F.softmax(out, dim=1).cpu())

    return torch.stack(preds).mean(dim=0)

def evaluate_tta(model, loader, device, n_augment=3):
    """Accuracy [%] with Test‑Time Augmentation."""
    correct = total = 0
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            prob = tta_predict(model, images, device, n_augment)
            _, pred = torch.max(prob, 1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total

def evaluate_no_tta(model, loader, device):
    """Vanilla single‑pass accuracy [%]."""
    correct = total = 0
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                out = model(images)
            _, pred = torch.max(out, 1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model = MiniXception(num_classes=7).to(device)
    best_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    best_model.eval()

    acc_plain = evaluate_no_tta(best_model, test_loader, device)
    acc_tta   = evaluate_tta(best_model, test_loader, device, n_augment=3)

    print(f"Test accuracy (plain) : {acc_plain:.2f}%")
    print(f"Test accuracy (TTA x3): {acc_tta:.2f}%")
