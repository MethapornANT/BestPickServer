import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import glob

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
                    img_path = os.path.join(label_dir, fname)
                    # ตรวจสอบว่าไฟล์รูปภาพสามารถเปิดได้หรือไม่
                    try:
                        with Image.open(img_path) as img:
                            img.verify()  # ตรวจสอบว่าไฟล์ไม่เสีย
                        self.samples.append((img_path, label2idx[label]))
                    except Exception as e:
                        print(f"ข้ามไฟล์ที่เสีย: {img_path} - {str(e)}")
                        continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"ไม่สามารถโหลดรูปภาพ: {img_path} - {str(e)}")
            # สร้างรูปภาพสีดำขนาด 224x224 เป็น fallback
            fallback_image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                fallback_image = self.transform(fallback_image)
            return fallback_image, label

# ================== TRANSFORMS ==================
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader

# ================== MODEL ==================
def get_model(num_classes):
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, num_classes)
    )
    return model

# ================== TRAIN ==================
def train_model(model, train_loader, val_loader, device, epochs=30, lr=1e-5, patience=8):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    best_acc = 0
    best_epoch = 0
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
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
        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping: หยุด train ที่ epoch {epoch+1} เพราะ val acc ไม่ดีขึ้นต่อเนื่อง {patience} รอบ")
            break
    print("Training complete. Best val acc: {:.4f} (epoch {})".format(best_acc, best_epoch+1))
    print("โหลด model ที่ val acc ดีที่สุด (epoch {}) ไปใช้ต่อ".format(best_epoch+1))

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

# ================== SAVE RESULT IMAGES (PREDICT ALL data) ==================
def save_result_images_data(model, device, result_dir='./result', diff_dir='./Different'):
    model.eval()
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(diff_dir):
        os.makedirs(diff_dir)
    for label in LABELS:
        os.makedirs(os.path.join(result_dir, label), exist_ok=True)
    data_dir = './data'
    all_images = []
    for category in LABELS:
        all_images.extend([(img_path, category) for img_path in glob.glob(os.path.join(data_dir, category, '*.jpg'))])
        all_images.extend([(img_path, category) for img_path in glob.glob(os.path.join(data_dir, category, '*.jpeg'))])
        all_images.extend([(img_path, category) for img_path in glob.glob(os.path.join(data_dir, category, '*.png'))])
    for img_path, true_label in tqdm(all_images, desc='Predicting data'):
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform_eval(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(image_tensor)
                pred = torch.argmax(output, dim=1).item()
                pred_label = idx2label[pred]
            # Copy ไป result/ ตาม label ที่โมเดลทำนาย
            dest_dir = os.path.join(result_dir, pred_label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(img_path, dest_dir)
            # ถ้า label เดิมกับ label ทำนายไม่ตรงกัน ให้ copy ไป Different/เดิมtoใหม่/
            if pred_label != true_label:
                diff_folder = f"{true_label.capitalize()}to{pred_label.capitalize()}"
                diff_path = os.path.join(diff_dir, diff_folder)
                os.makedirs(diff_path, exist_ok=True)
                shutil.copy2(img_path, diff_path)
        except Exception as e:
            print(f"ข้ามไฟล์ {img_path}: {e}")

# ================== MAIN ==================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    model = get_model(num_classes=len(LABELS)).to(device)
    train_model(model, train_loader, val_loader, device, epochs=30, lr=1e-5, patience=2)
    print("\n=== Evaluation on Test Set ===")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    evaluate(model, test_loader, device, print_report=True)
    print("\n=== Predicting all data and saving to result/ & Different/ ===")
    save_result_images_data(model, device, result_dir='./result', diff_dir='./Different')
    print("บันทึกรูปภาพที่ทำนายเสร็จแล้วในโฟลเดอร์ result/ (แบ่งตาม label ที่โมเดลทำนาย) และ Different/ (เฉพาะภาพที่ label เปลี่ยน)")

# ========== หมายเหตุ ===========
# - สร้างโฟลเดอร์ nsfw_data/train, nsfw_data/val, nsfw_data/test
#   และแยก subfolder ตาม label (normal, hentai, porn, sexy, anime)
# - ใส่รูปแต่ละประเภทใน subfolder ให้ตรง label
# - สามารถเพิ่ม class label ได้ตามต้องการ (แก้ LABELS ด้านบน)
# - สามารถปรับ backbone เป็น resnet50, efficientnet, swin ฯลฯ ได้ 