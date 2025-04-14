import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.distributions import Beta
from model import MiniXception
from utils import FocalLoss, SubsetWithTransform

# Cesty a hyperparametre
TRAIN_PATH = '/content/dataset/train'
MODEL_PATH = 'best_model.pt'
LEARNING_RATE = 1e-3
W_DECAY = 1e-4
BATCH_SIZE = 32
EPOCHS = 300
PATIENCE = 30

# Dataset a transformácie
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
full_dataset = torchvision.datasets.ImageFolder(root=TRAIN_PATH, transform=None)

# Rozdelíme train / val
train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

num_classes = len(full_dataset.classes)
print("Triedy:", full_dataset.classes)

# Transform
train_transform = T.Compose([
    T.Grayscale(),
    T.RandomResizedCrop(128, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

val_transform = T.Compose([
    T.Grayscale(),
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

train_dataset = SubsetWithTransform(train_subset, train_transform)
val_dataset   = SubsetWithTransform(val_subset, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Tréningová funkcia
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=EPOCHS, patience=PATIENCE):
    best_val_acc = 0.0
    no_improve = 0

    # Mixup
    alpha = 0.2
    beta_dist = Beta(alpha, alpha)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Mixup
            lam = beta_dist.sample().to(device)
            idx = torch.randperm(images.size(0), device=device)
            mixed = lam * images + (1 - lam) * images[idx]
            labels_a, labels_b = labels, labels[idx]

            optimizer.zero_grad()

            with autocast():
                outputs = model(mixed)
                loss_a = criterion(outputs, labels_a)
                loss_b = criterion(outputs, labels_b)
                loss   = lam * loss_a + (1 - lam) * loss_b

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        train_loss = running_loss / total_samples

        # Validácia
        model.eval()
        val_correct = 0
        val_total   = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outs = model(images)
                    val_loss_sum += criterion(outs, labels).item() * images.size(0)
                _, preds = torch.max(outs, 1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc  = 100.0 * val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("Model uložený!")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping na epoche {epoch + 1}")
                break

        print(f"Epcha {epoch + 1}/{epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.2f}%")

    print(f"Koniec tréningu. Najlepšia val acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    model = MiniXception(num_classes=num_classes).to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=W_DECAY)
    scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    scaler = GradScaler()
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device)
