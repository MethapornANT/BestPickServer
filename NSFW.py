import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import glob
import pandas as pd
import time
import logging
import warnings

# ================== CONFIG ==================
TRAIN_DIR = './nsfw_data/train'
VAL_DIR = './nsfw_data/val'
TEST_DIR = './nsfw_data/test'
DATA_DIR = './data'

LABELS = ['normal', 'hentai', 'porn', 'sexy', 'anime']
label2idx = {label: idx for idx, label in enumerate(LABELS)}
idx2label = {idx: label for label, idx in label2idx.items()}

MODEL_NAMES = ['ResNet50', 'EfficientNetB0', 'ViT-B_16']

CHECKPOINT_DIR = './checkpoints'# โฟลเดอร์สำหรับเก็บ checkpoint
PAUSE_SIGNAL_FILE = 'PAUSE_SIGNAL.txt'# ไฟล์สำหรับส่งสัญญาณหยุดชั่วคราว
STOP_SIGNAL_FILE = 'STOP_SIGNAL.txt'# ไฟล์สำหรับส่งสัญญาณหยุดถาวร

# ================== LOGGING SETUP ==================
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# ================== SUPPRESS WARNINGS ==================
warnings.filterwarnings("ignore", category=UserWarning, module='PIL.Image')

# ================== HYPERPARAMETERS FOR EACH MODEL ==================
def get_model_hyperparameters(model_name):
    """
    Returns a dictionary of hyperparameters for a given model name.
    """
    if model_name == 'ResNet50':
        return {
            'epochs': 50,
            'lr': 0.00003,
            'patience': 3,
            'scheduler_patience': 2,
            'batch_size': 64
        }
    elif model_name == 'EfficientNetB0':
        return {
            'epochs': 50,
            'lr': 0.00003,
            'patience': 3,
            'scheduler_patience': 2,
            'batch_size': 64
        }
    elif model_name == 'ViT-B_16':
        return {
            'epochs': 50,
            'lr': 0.00003,
            'patience': 3,
            'scheduler_patience': 2,
            'batch_size': 32
        }
    else:
        raise ValueError(f"No hyperparameters defined for model: {model_name}")

# ================== DATASET ==================
class NSFWImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        
        logger.info(f"Loading data from: {root_dir}")
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(label_dir, fname)
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        if label in label2idx:
                            self.samples.append((img_path, label2idx[label]))
                        else:
                            logger.warning(f"Skipping file {img_path}: Label '{label}' not found in LABELS.")
                    except Exception as e:
                        logger.warning(f"Skipping corrupted file: {img_path} - {str(e)}")
                        continue
        logger.info(f"Data loading complete: Found {len(self.samples)} images in {root_dir}")

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
            logger.warning(f"Failed to load image: {img_path} - {str(e)}. Using fallback black image.")
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
    logger.info("Preparing DataLoaders...")
    train_ds = NSFWImageDataset(TRAIN_DIR, transform=transform_train)
    val_ds = NSFWImageDataset(VAL_DIR, transform=transform_eval)
    test_ds = NSFWImageDataset(TEST_DIR, transform=transform_eval)
    
    num_workers_to_use = os.cpu_count() // 2 
    if num_workers_to_use == 0:
        num_workers_to_use = 1 
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers_to_use)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers_to_use)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers_to_use)
    logger.info("DataLoaders preparation complete.")
    return train_loader, val_loader, test_loader

# ================== MODEL ==================
def get_model(model_name, num_classes):
    logger.info(f"Loading model: {model_name} with IMAGENET1K_V1 weights.")
    model = None
    if model_name == 'ResNet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_name == 'EfficientNetB0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(model.classifier[1].in_features, num_classes)
        )
    elif model_name == 'ViT-B_16':
        model = models.vit_b_16(weights='IMAGENET1K_V1')
        model.heads = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(model.heads.head.in_features, num_classes)
        )
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
    logger.info(f"Model {model_name} loaded successfully.")
    return model

# ================== CHECKPOINTING ==================
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved to: {filepath}")

def load_checkpoint(model, optimizer, scheduler, filename="checkpoint.pth.tar", force_start_from_scratch=False):
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    
    # กำหนดค่าเริ่มต้น
    start_epoch = 0
    best_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_acc': []}

    if force_start_from_scratch or not os.path.isfile(filepath):
        logger.info("No checkpoint found or 'Start from Scratch' selected. Starting training from scratch.")
        return start_epoch, best_acc, patience_counter, history
    else:
        logger.info(f"Loading checkpoint from: {filepath}")
        try:
            # โหลด checkpoint เข้า CPU ก่อนเพื่อความปลอดภัย แล้วค่อยย้ายไป device ทีหลัง
            checkpoint = torch.load(filepath, map_location=torch.device('cpu')) 
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            
            # ดึงค่าจาก checkpoint
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            patience_counter = checkpoint['patience_counter']
            
            # --- แก้ไขตรงนี้: โหลด history มา แต่จะถูกเคลียร์ถ้า resume เพื่อให้แน่ใจว่าไม่ซ้ำ ---
            # เราจะบันทึก history ใหม่ใน train_model จาก epoch ที่เริ่มต้นใหม่นี้
            # ดังนั้นจึงไม่จำเป็นต้องนำ history เก่ามาใช้ตรงๆ เพราะจะถูก append ใหม่ในลูป
            # ถ้าอยากให้ history รวมทุก epoch ตั้งแต่แรกจริงๆ จะต้องนำมา concat กัน
            # แต่เพื่อความง่ายและไม่ให้ซ้ำซ้อนใน CSV (ซึ่งจะถูกสร้างใหม่ทุกครั้งที่รัน)
            # เราจะถือว่า history ที่โหลดมาเป็นเพียงค่าอ้างอิงสถานะล่าสุด (best_acc)
            # ส่วนข้อมูล history ที่จะเขียนลง CSV ใหม่จะเริ่มเก็บจาก epoch ที่โหลดมา
            # ดังนั้นตรงนี้เราจะให้ history เป็นค่าว่างเริ่มต้นเสมอเมื่อโหลด checkpoint
            # และจะให้ train_model บันทึกใหม่ตั้งแต่ epoch ที่ resume
            # หรืออีกทางคือ ถ้าต้องการรวม history เก่าจริงๆ ต้องทำแบบนี้:
            history = checkpoint.get('history', {'train_loss': [], 'val_acc': []})
            # อันนี้คือการคง history เดิมไว้ ซึ่งจะถูก append ต่อไป
            # ดังนั้นเมื่อเริ่มต้น Epoch ที่ 1 (ถ้าโหลด checkpoint ที่จบ Epoch 0)
            # ไฟล์ CSV จะมีข้อมูล Epoch 0-X

            logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch + 1} with best_acc {best_acc:.4f}.")
            return start_epoch, best_acc, patience_counter, history
        except Exception as e:
            logger.warning(f"Error loading checkpoint {filepath}: {e}. Starting training from scratch.")
            return 0, 0, 0, {'train_loss': [], 'val_acc': []}


# ================== TRAIN ==================
def train_model(model, train_loader, val_loader, device, model_save_path, epochs, lr, patience, scheduler_patience, checkpoint_filename="checkpoint.pth.tar", force_start_from_scratch=False):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=scheduler_patience, factor=0.5)
    
    start_epoch, best_acc, patience_counter, history = load_checkpoint(
        model, optimizer, scheduler, checkpoint_filename, force_start_from_scratch
    )
    
    model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    # กำหนด initial_best_acc เพื่อใช้ในการหา final_best_acc_from_history
    # และ best_epoch สำหรับการแสดงผลสุดท้าย
    initial_best_acc = best_acc 
    # ถ้า history มีข้อมูลอยู่แล้ว (จากการโหลด checkpoint) ให้หา best_epoch ที่เจอใน history เก่า
    if history['val_acc']:
        initial_best_epoch_idx = np.argmax(history['val_acc'])
        final_best_epoch_overall = initial_best_epoch_idx + (start_epoch - len(history['val_acc']) if len(history['val_acc']) > 0 and start_epoch > 0 else 0)
    else:
        final_best_epoch_overall = 0 # ถ้าไม่มี history ให้เริ่มต้นเป็น epoch 0
    
    start_time_overall = time.time() # เวลาเริ่มการรันครั้งนี้ทั้งหมด

    for epoch in range(start_epoch, epochs): 
        # ตรวจสอบไฟล์หยุดถาวร
        if os.path.exists(STOP_SIGNAL_FILE):
            logger.info(f"STOP_SIGNAL detected. Terminating training for {model.__class__.__name__}.")
            os.remove(STOP_SIGNAL_FILE) # ลบไฟล์สัญญาณออก
            save_checkpoint({ # บันทึก checkpoint สุดท้ายก่อนหยุดถาวร
                'epoch': epoch, # บันทึก epoch ที่หยุด (epoch ที่กำลังจะเริ่มเทรน)
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
                'patience_counter': patience_counter,
                'history': history,
            }, filename=checkpoint_filename)
            logger.info(f"Training terminated by STOP_SIGNAL for {model.__class__.__name__}.")
            # คำนวณเวลาที่ใช้ไปจนถึงจุดที่หยุด
            train_time_taken = time.time() - start_time_overall
            return best_acc, history, train_time_taken # จบการทำงาน
            
        model.train()
        running_loss = 0
        
        # ปรับการแสดงผลของ tqdm ให้แสดง epoch ที่ถูกต้องตามที่กำลังเทรน
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} (Train)", unit="batch")
        for images, labels in pbar:
            # ตรวจสอบไฟล์หยุดชั่วคราวในระหว่าง batch 
            if os.path.exists(PAUSE_SIGNAL_FILE):
                logger.info(f"PAUSE_SIGNAL detected. Training paused for {model.__class__.__name__} at Epoch {epoch+1}, Batch {pbar.n+1}.")
                save_checkpoint({
                    'epoch': epoch, # บันทึก epoch ที่กำลังจะรัน
                    'batch_idx': pbar.n, # บันทึก batch ที่หยุดไว้
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_acc': best_acc,
                    'patience_counter': patience_counter,
                    'history': history,
                }, filename=checkpoint_filename)
                
                while os.path.exists(PAUSE_SIGNAL_FILE):
                    logger.info("Training is paused. Please remove PAUSE_SIGNAL.txt to resume...")
                    time.sleep(5) # รอ 5 วินาที
                logger.info(f"PAUSE_SIGNAL cleared. Resuming training for {model.__class__.__name__} from Epoch {epoch+1}.")

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=running_loss/((pbar.n + 1) * images.size(0)), refresh=True)
            
        avg_train_loss = running_loss / len(train_loader.dataset)
        val_acc, avg_val_loss, _ = evaluate(model, val_loader, device, print_report=False)
        
        # history จะถูกบันทึกสำหรับ epoch ปัจจุบัน
        history['train_loss'].append(avg_train_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            # อัปเดต final_best_epoch_overall ให้เป็น epoch ปัจจุบัน (ที่ +1 เพราะ epoch นับจาก 0)
            final_best_epoch_overall = epoch + 1 
            torch.save(model.state_dict(), model_save_path) # บันทึก best model
            patience_counter = 0
            logger.info(f"Best model saved (Val Acc: {best_acc:.4f}).")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping: Training stopped at epoch {epoch+1} as validation accuracy did not improve for {patience} consecutive epochs.")
            break
            
        # บันทึก checkpoint หลังจบแต่ละ Epoch
        save_checkpoint({
            'epoch': epoch + 1, # บันทึก epoch ถัดไปที่จะเริ่ม
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_acc': best_acc,
            'patience_counter': patience_counter,
            'history': history, # บันทึก history ทั้งหมด
        }, filename=checkpoint_filename)

    end_time_overall = time.time()
    train_time_taken = end_time_overall - start_time_overall
    
    # ใช้ final_best_epoch_overall ที่อัปเดตไปแล้ว
    # ถ้าไม่มีการ improve เลย (เช่น early stopping ทันที) อาจจะต้องย้อนกลับไปดู best_acc ใน history
    # แต่ด้วย logic ที่อัปเดต best_acc และ final_best_epoch_overall พร้อมกัน
    # ค่าเหล่านี้ควรจะถูกต้องเสมอ
    
    logger.info(f"Training complete. Best Val Acc: {best_acc:.4f} (epoch {final_best_epoch_overall}).")
    logger.info(f"Loading best model (from epoch {final_best_epoch_overall}) for further use.")
    
    # ส่งคืนค่า best_acc ที่ได้จากการเทรนทั้งหมด (รวมที่ resume มา)
    # และ history ทั้งหมด (รวมที่โหลดมา)
    # และเวลาที่ใช้ในการรันครั้งนี้
    return best_acc, history, train_time_taken

# ================== EVALUATE ==================
def evaluate(model, loader, device, print_report=True):
    model.eval()
    all_preds = []
    all_labels = []
    
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc="Evaluating...", unit="batch")
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(loader.dataset)

    report = None
    if print_report:
        logger.info(f"=== Classification Report ===")
        report = classification_report(all_labels, all_preds, target_names=LABELS, output_dict=True)
        logger.info("\n" + classification_report(all_labels, all_preds, target_names=LABELS))
        
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8,6))
        plt.imshow(cm, cmap='Blues')
        plt.title(f'Confusion Matrix - {model.__class__.__name__}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(np.arange(len(LABELS)), LABELS, rotation=45)
        plt.yticks(np.arange(len(LABELS)), LABELS)
        plt.colorbar()
        plt.tight_layout()

        # แก้ไขตรงนี้ให้บันทึกรูปแทนการแสดง
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        figure_filename = f'confusion_matrix_{model.__class__.__name__}_{timestamp}.png'
        
        output_dir = './evaluation_plots'
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(os.path.join(output_dir, figure_filename))
        logger.info(f"Confusion Matrix saved to: {os.path.join(output_dir, figure_filename)}")
        plt.close() # ปิด figure เพื่อไม่ให้รูปภาพสะสมในหน่วยความจำ
    return acc, avg_loss, report

# ================== PREDICT AND SAVE RESULT IMAGES ==================
def predict_and_save_results(model, device, model_name, result_dir_base='./result', diff_dir_base='./Different'):
    model.eval()
    
    result_dir = os.path.join(result_dir_base, model_name)
    diff_dir = os.path.join(diff_dir_base, model_name)

    logger.info(f"Preparing folders for {model_name} results...")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(diff_dir, exist_ok=True)

    for label in LABELS:
        os.makedirs(os.path.join(result_dir, label), exist_ok=True)
    logger.info("Folder preparation complete.")
    
    all_images_paths = []
    all_true_labels = []
    
    logger.info(f"Collecting all images from {DATA_DIR} for prediction...")
    for category in LABELS:
        current_data_dir = os.path.join(DATA_DIR, category)
        if os.path.isdir(current_data_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in glob.glob(os.path.join(current_data_dir, ext)):
                    all_images_paths.append(img_path)
                    all_true_labels.append(category)
        else:
            logger.warning(f"Folder '{current_data_dir}' does not exist and will be skipped.")
    total_images = len(all_images_paths)
    logger.info(f"Collected {total_images} images.")

    correct_predictions = 0
    
    actual_labels_for_report = []
    predicted_labels_for_report = []

    for i, img_path in tqdm(enumerate(all_images_paths), total=total_images, desc=f'Predicting & Saving for {model_name}', unit="image"):
        true_label = all_true_labels[i]
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform_eval(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(image_tensor)
                pred = torch.argmax(output, dim=1).item()
                pred_label = idx2label[pred]
            
            dest_dir = os.path.join(result_dir, pred_label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(img_path, dest_dir)
            
            if pred_label != true_label:
                diff_folder = f"{true_label.capitalize()}to{pred_label.capitalize()}"
                diff_path = os.path.join(diff_dir, diff_folder)
                os.makedirs(diff_path, exist_ok=True)
                shutil.copy2(img_path, diff_path)
            else:
                correct_predictions += 1
            
            actual_labels_for_report.append(true_label)
            predicted_labels_for_report.append(pred_label)

        except Exception as e:
            logger.warning(f"Skipping file {img_path}: {e}")
            
    overall_accuracy = correct_predictions / total_images if total_images > 0 else 0
    logger.info(f"Overall prediction accuracy for all images in {DATA_DIR} with {model_name}: {overall_accuracy:.4f}")

    report_all_data = None
    if actual_labels_for_report and predicted_labels_for_report:
        logger.info(f"\nClassification Report for all data prediction in {DATA_DIR} with {model_name}:")
        report_all_data = classification_report(actual_labels_for_report, predicted_labels_for_report, target_names=LABELS, output_dict=True)
        logger.info("\n" + classification_report(actual_labels_for_report, predicted_labels_for_report, target_names=LABELS))

    return overall_accuracy, report_all_data

# ================== MAIN ==================
if __name__ == '__main__':
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Please check your PyTorch and NVIDIA Driver installation.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running on Device: {device}")

    results = [] 

    print("\n----- Training Options -----")
    print("1. Start new training (delete old checkpoints/models)")
    print("2. Start new training (rename old checkpoints/models)")
    print("3. Resume training from last checkpoint")

    choice = input("Please choose an option (1/2/3): ")

    force_start_from_scratch_flag = False
    
    if choice == '1':
        if os.path.exists(CHECKPOINT_DIR):
            shutil.rmtree(CHECKPOINT_DIR)
            logger.info(f"Removed existing checkpoint directory: {CHECKPOINT_DIR}")
        for model_name in MODEL_NAMES:
            model_base_name = model_name.lower().replace("-", "_")
            best_model_file = f'./{model_base_name}_best.pth'
            if os.path.exists(best_model_file):
                os.remove(best_model_file)
                logger.info(f"Removed existing best model file: {best_model_file}")
        force_start_from_scratch_flag = True
        logger.info("Option 1 selected: Starting new training from scratch, old files removed.")

    elif choice == '2':
        timestamp_for_rename = time.strftime("%Y%m%d-%H%M%S")
        if os.path.exists(CHECKPOINT_DIR):
            new_checkpoint_dir = f"{CHECKPOINT_DIR}_old_{timestamp_for_rename}"
            os.rename(CHECKPOINT_DIR, new_checkpoint_dir)
            logger.info(f"Renamed existing checkpoint directory to: {new_checkpoint_dir}")
        for model_name in MODEL_NAMES:
            model_base_name = model_name.lower().replace("-", "_")
            best_model_file = f'./{model_base_name}_best.pth'
            if os.path.exists(best_model_file):
                new_best_model_file = f'./{model_base_name}_best_old_{timestamp_for_rename}.pth'
                os.rename(best_model_file, new_best_model_file)
                logger.info(f"Renamed existing best model file to: {new_best_model_file}")
        force_start_from_scratch_flag = True
        logger.info("Option 2 selected: Starting new training from scratch, old files renamed.")

    elif choice == '3':
        force_start_from_scratch_flag = False
        logger.info("Option 3 selected: Resuming training from last checkpoint.")
    else:
        logger.warning("Invalid choice. Defaulting to 'Resume training from last checkpoint'.")
        force_start_from_scratch_flag = False

    # ดึง timestamp สำหรับใช้ในชื่อไฟล์ CSV หลัก (ให้เป็น timestamp ของการรันครั้งนี้)
    current_run_timestamp = time.strftime("%Y%m%d-%H%M%S")

    for model_name_to_train in MODEL_NAMES:
        checkpoint_filename = f'{model_name_to_train.lower().replace("-", "_")}_checkpoint.pth.tar'
        model_save_path = f'./{model_name_to_train.lower().replace("-", "_")}_best.pth'

        model_hparams = get_model_hyperparameters(model_name_to_train)
        epochs_for_model = model_hparams['epochs']
        lr_for_model = model_hparams['lr']
        patience_for_model = model_hparams['patience']
        scheduler_patience_for_model = model_hparams['scheduler_patience']
        batch_size_for_model = model_hparams['batch_size']

        logger.info(f"\n{'='*50}")
        logger.info(f"===== Starting model training: {model_name_to_train} =====")
        logger.info(f"Hyperparameters: Epochs={epochs_for_model}, LR={lr_for_model}, Patience={patience_for_model}, Scheduler Patience={scheduler_patience_for_model}, Batch Size={batch_size_for_model}")
        logger.info(f"{'='*50}")
        
        model = get_model(model_name_to_train, num_classes=len(LABELS))
        
        logger.info(f"Initiating training for {model_name_to_train}...")
        
        # สร้าง DataLoader ใหม่สำหรับแต่ละโมเดล ด้วย batch_size ที่กำหนดไว้
        train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size_for_model)

        best_val_acc, history, train_time_taken = train_model(
            model, train_loader, val_loader, device, model_save_path, 
            epochs=epochs_for_model, 
            lr=lr_for_model, 
            patience=patience_for_model, 
            scheduler_patience=scheduler_patience_for_model,
            checkpoint_filename=checkpoint_filename,
            force_start_from_scratch=force_start_from_scratch_flag
        )
        
        # --- เริ่มส่วนเพิ่มโค้ดสำหรับบันทึก Training History ---
        if history['train_loss'] and history['val_acc']:
            # สร้าง DataFrame โดยที่ Epoch จะเริ่มจาก 1 เสมอ
            history_df = pd.DataFrame({
                'Epoch': range(1, len(history['train_loss']) + 1), 
                'Train Loss': history['train_loss'],
                'Val Accuracy': history['val_acc']
            })
            # ใช้ current_run_timestamp ที่ดึงมาจากตอนต้น main
            history_csv_filename = f'./{model_name_to_train.lower().replace("-", "_")}_training_history_{current_run_timestamp}.csv'
            history_df.to_csv(history_csv_filename, index=False)
            logger.info(f"Training history for {model_name_to_train} saved to: {history_csv_filename}")
        else:
            logger.warning(f"No training history data available for {model_name_to_train} to save.")
        # --- จบส่วนเพิ่มโค้ดสำหรับบันทึก Training History ---
        
        logger.info(f"\n--- Evaluation of {model_name_to_train} on Test Set ---")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.to(device) 
        test_acc, test_loss, test_report = evaluate(model, test_loader, device, print_report=True)
        
        logger.info(f"\n--- Predicting all data in '{DATA_DIR}' with {model_name_to_train} ---")
        all_data_accuracy, all_data_report = predict_and_save_results(
            model, device, model_name_to_train, result_dir_base='./result', diff_dir_base='./Different'
        )
        logger.info(f"Predicted images for {model_name_to_train} saved in result/{model_name_to_train}/ and Different/{model_name_to_train}/.")

        model_results = {
            'Model': model_name_to_train,
            'Best Validation Accuracy': best_val_acc,
            'Test Accuracy': test_acc,
            'Test Loss': test_loss,
            'Total Training Time (seconds)': train_time_taken, # ใช้ train_time_taken ที่ส่งกลับมา
            'All Data Prediction Accuracy': all_data_accuracy,
        }
        
        if test_report:
            for label_name in LABELS:
                if label_name in test_report:
                    model_results[f'Test Precision ({label_name})'] = test_report[label_name]['precision']
                    model_results[f'Test Recall ({label_name})'] = test_report[label_name]['recall']
                    model_results[f'Test F1-Score ({label_name})'] = test_report[label_name]['f1-score']
            model_results['Test Macro Avg Precision'] = test_report['macro avg']['precision']
            model_results['Test Macro Avg Recall'] = test_report['macro avg']['recall']
            model_results['Test Macro Avg F1-Score'] = test_report['macro avg']['f1-score']
            model_results['Test Weighted Avg Precision'] = test_report['weighted avg']['precision']
            model_results['Test Weighted Avg Recall'] = test_report['weighted avg']['recall']
            model_results['Test Weighted Avg F1-Score'] = test_report['weighted avg']['f1-score']

        if all_data_report:
            for label_name in LABELS:
                if label_name in all_data_report:
                    model_results[f'All Data Precision ({label_name})'] = all_data_report[label_name]['precision']
                    model_results[f'All Data Recall ({label_name})'] = all_data_report[label_name]['recall']
                    model_results[f'All Data F1-Score ({label_name})'] = all_data_report[label_name]['f1-score']
            model_results['All Data Macro Avg Precision'] = all_data_report['macro avg']['precision']
            model_results['All Data Macro Avg Recall'] = all_data_report['macro avg']['recall']
            model_results['All Data Macro Avg F1-Score'] = all_data_report['macro avg']['f1-score']
            model_results['All Data Weighted Avg Precision'] = all_data_report['weighted avg']['precision']
            model_results['All Data Weighted Avg Recall'] = all_data_report['weighted avg']['recall']
            model_results['All Data Weighted Avg F1-Score'] = all_data_report['weighted avg']['f1-score']
            
        results.append(model_results)

    results_df = pd.DataFrame(results)
    csv_filename = f'./model_comparison_results_{current_run_timestamp}.csv'
    results_df.to_csv(csv_filename, index=False)
    logger.info(f"\n--- All model comparisons completed! ---")
    logger.info(f"Summary results saved to: {csv_filename}")

    logger.info("\nSummary of Model Comparisons:")
    logger.info(results_df.T.to_string())