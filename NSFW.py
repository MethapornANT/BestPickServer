import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# ================== CONFIG ==================
# กำหนด path สำหรับ dataset
TRAIN_DIR = './nsfw_data/train'      # โฟลเดอร์สำหรับรูป train (subfolder: normal, hentai, porn, etc.)
VAL_DIR = './nsfw_data/val'          # โฟลเดอร์สำหรับรูป validation
TEST_DIR = './nsfw_data/test'        # โฟลเดอร์สำหรับรูป test/evaluation
MODEL_SAVE_PATH = './nsfw_model_best.pth'

# กำหนด label mapping (แก้ไขตาม class ที่มี)
LABELS = ['normal', 'hentai', 'porn', 'sexy', 'anime']
label2idx = {label: idx for idx, label in enumerate(LABELS)}
idx2label = {idx: label for label, idx in label2idx.items()}

# ================== DATASET ==================
class NSFWImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(label_dir, fname), label2idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ================== TRANSFORMS ==================
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ================== LOAD DATA ==================
def get_dataloaders(batch_size=32):
    train_ds = NSFWImageDataset(TRAIN_DIR, transform=transform_train)
    val_ds = NSFWImageDataset(VAL_DIR, transform=transform_eval)
    test_ds = NSFWImageDataset(TEST_DIR, transform=transform_eval)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader

# ================== MODEL ==================
def get_model(num_classes):
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

# ================== TRAIN ==================
def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device, print_report=False)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Training complete. Best val acc: {:.4f}".format(best_acc))

# ================== EVALUATE ==================
def evaluate(model, loader, device, print_report=True):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    if print_report:
        print(classification_report(all_labels, all_preds, target_names=LABELS))
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8,6))
        plt.imshow(cm, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(np.arange(len(LABELS)), LABELS, rotation=45)
        plt.yticks(np.arange(len(LABELS)), LABELS)
        plt.colorbar()
        plt.show()
    return acc

# ================== MAIN ==================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    model = get_model(num_classes=len(LABELS)).to(device)
    train_model(model, train_loader, val_loader, device, epochs=15, lr=1e-4)
    print("\n=== Evaluation on Test Set ===")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    evaluate(model, test_loader, device, print_report=True)

# ========== หมายเหตุ ===========
# - สร้างโฟลเดอร์ nsfw_data/train, nsfw_data/val, nsfw_data/test
#   และแยก subfolder ตาม label (normal, hentai, porn, sexy, anime)
# - ใส่รูปแต่ละประเภทใน subfolder ให้ตรง label
# - สามารถเพิ่ม class label ได้ตามต้องการ (แก้ LABELS ด้านบน)
# - สามารถปรับ backbone เป็น resnet50, efficientnet, swin ฯลฯ ได้ 