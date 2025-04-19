"""
Script that evaluates a saved MiniXception checkpoint on the test set,
once directly and once with simple test‑time augmentation (random flips).
"""

import sys
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from model import MiniXception

TEST_PATH = "../data/test"
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "../results/best_model.pt"

# --------------------------------------------------------------------------- #
# 1. Dataset / transforms                                                     #
# --------------------------------------------------------------------------- #
test_transform = T.Compose(
    [
        T.Grayscale(),
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
)

test_dataset = torchvision.datasets.ImageFolder(root=TEST_PATH, transform=test_transform)
test_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)

# --------------------------------------------------------------------------- #
# 2. Inference helpers                                                        #
# --------------------------------------------------------------------------- #
def tta_predict(model, img, device, n_augment=3):
    """
    Run *n_augment* horizontal‑flip variants and return the averaged probs.
    """
    base = img[0].cpu()  # C × H × W
    preds = []

    for _ in range(n_augment):
        # 50 % chance to flip; otherwise leave unchanged
        aug = torch.flip(base, dims=[2]) if torch.rand(1).item() < 0.5 else base
        aug = aug.unsqueeze(0).to(device)

        with autocast():
            out = model(aug)
        preds.append(F.softmax(out, dim=1).cpu())

    return torch.stack(preds).mean(dim=0)


def evaluate_tta(model, loader, device, n_augment=3):
    """Accuracy with test‑time augmentation."""
    correct = total = 0
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            prob = tta_predict(model, images, device, n_augment)
            pred = torch.argmax(prob, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def evaluate_no_tta(model, loader, device):
    """Plain single‑crop accuracy."""
    correct = total = 0
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                out = model(images)
            pred = torch.argmax(out, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


# --------------------------------------------------------------------------- #
# 3. Main — load checkpoint, report metrics                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniXception(num_classes=7).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    acc_plain = evaluate_no_tta(model, test_loader, device)
    acc_tta = evaluate_tta(model, test_loader, device, n_augment=3)

    print(f"Test accuracy (no TTA): {acc_plain:.2f}%")
    print(f"Test accuracy (TTA):    {acc_tta:.2f}%")
