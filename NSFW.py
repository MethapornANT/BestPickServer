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
TRAIN_DIR = './nsfw_data2/train'
VAL_DIR = './nsfw_data2/val'
TEST_DIR = './nsfw_data2/test'

LABELS = ['normal', 'hentai', 'porn', 'sexy', 'anime']
label2idx = {label: idx for idx, label in enumerate(LABELS)}
idx2label = {idx: label for label, idx in label2idx.items()}

MODEL_NAMES = ['ResNet18', 'EfficientNetB0', 'MobileNetV3_Small']

BASE_OUTPUT_DIR = './NSFW_Model'
RESULTS_BASE_DIR = os.path.join(BASE_OUTPUT_DIR, 'results')
CHECKPOINT_DIR = os.path.join(BASE_OUTPUT_DIR, 'checkpoints')

PAUSE_SIGNAL_FILE = os.path.join(BASE_OUTPUT_DIR, 'PAUSE_SIGNAL.txt')
STOP_SIGNAL_FILE = os.path.join(BASE_OUTPUT_DIR, 'STOP_SIGNAL.txt')

# ================== LOGGING SETUP ==================
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.handlers:
    for handler in logger.handlers:
        logger.removeHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(console_handler)

# ================== SUPPRESS WARNINGS ==================
warnings.filterwarnings("ignore", category=UserWarning, module='PIL.Image')

# ================== HYPERPARAMETERS FOR EACH MODEL ==================
def get_model_hyperparameters(model_name):
    """
    Returns a dictionary of hyperparameters for a given model name.
    """
    common_hparams = {
        'epochs': 100,
        'lr': 0.0001,
        'patience': 7,
        'scheduler_patience': 3,
        'batch_size': 128
    }

    if model_name == 'MobileNetV3_Small':
        common_hparams['batch_size'] = 64
    
    return common_hparams

# ================== DATASET ==================
class NSFWImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        
        print(f"Loading data from: {root_dir}")
        if not os.path.exists(root_dir):
            logger.error(f"Error: Directory '{root_dir}' does not exist. Please run data splitting script first.")
            return

        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if not os.path.isdir(label_dir):
                continue
            if label not in label2idx:
                logger.warning(f"Skipping directory {label_dir}: Label '{label}' not found in LABELS list.")
                continue

            for fname in os.listdir(label_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(label_dir, fname)
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        self.samples.append((img_path, label2idx[label]))
                    except Exception as e:
                        logger.warning(f"Skipping corrupted or unreadable file: {img_path} - {str(e)}")
                        continue
        print(f"Data loading complete: Found {len(self.samples)} images in {root_dir}")

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
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.RandomRotation(15),
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
    print("Preparing DataLoaders...")
    train_ds = NSFWImageDataset(TRAIN_DIR, transform=transform_train)
    val_ds = NSFWImageDataset(VAL_DIR, transform=transform_eval)
    test_ds = NSFWImageDataset(TEST_DIR, transform=transform_eval)
    
    if not train_ds:
        logger.error(f"Train Dataset is empty. Please check '{TRAIN_DIR}' and data splitting process.")
        exit()
    if not val_ds:
        logger.warning(f"Validation Dataset is empty. This might affect early stopping. Please check '{VAL_DIR}'.")
    if not test_ds:
        logger.warning(f"Test Dataset is empty. Evaluation on test set will not be possible. Please check '{TEST_DIR}'.")
    
    num_workers_to_use = os.cpu_count() // 2 
    if num_workers_to_use == 0:
        num_workers_to_use = 1 
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers_to_use)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers_to_use)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers_to_use)
    print("DataLoaders preparation complete.")
    return train_loader, val_loader, test_loader

# ================== MODEL ==================
def get_model(model_name, num_classes):
    print(f"Loading model: {model_name} with appropriate weights.")
    model = None
    if model_name == 'ResNet18':
        model = models.resnet18(weights='IMAGENET1K_V1')
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
    elif model_name == 'MobileNetV3_Small': 
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        # แก้ไขตรงนี้! ดึง in_features จากเลเยอร์แรกสุดของ classifier เดิม
        first_classifier_in_features = model.classifier[0].in_features 
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2), # Standard dropout value
            torch.nn.Linear(first_classifier_in_features, num_classes)
        )
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
    
    if hasattr(torch, 'compile'):
        try:
            # model = torch.compile(model) 
            logger.info(f"Model {model_name} compile functionality available but currently skipped for stability.") 
        except Exception as e:
            logger.warning(f"Failed to compile model {model_name} with torch.compile: {e}. Running without compilation.")

    print(f"Model {model_name} loaded successfully.")
    return model

# ================== CHECKPOINTING ==================
def save_checkpoint(state, filename="checkpoint.pth.tar", checkpoint_dir=CHECKPOINT_DIR):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved to: {filepath}")

def load_checkpoint(model, optimizer, scheduler, filename="checkpoint.pth.tar", checkpoint_dir=CHECKPOINT_DIR, force_start_from_scratch=False):
    filepath = os.path.join(checkpoint_dir, filename)
    
    start_epoch = 0
    best_acc = 0
    patience_counter = 0
    # เพิ่ม val_loss และ epoch_times ใน history
    history = {'train_loss': [], 'val_acc': [], 'val_loss': [], 'epoch_times': []}
    total_training_time_overall = 0 # เพิ่มตัวแปรสำหรับเวลา training สะสม

    if force_start_from_scratch or not os.path.isfile(filepath):
        logger.info(f"No checkpoint found at {filepath} or 'Start from Scratch' selected. Starting training from scratch.")
        return start_epoch, best_acc, patience_counter, history, total_training_time_overall
    else:
        logger.info(f"Loading checkpoint from: {filepath}")
        try:
            checkpoint = torch.load(filepath, map_location=torch.device('cpu')) 
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            patience_counter = checkpoint['patience_counter']
            # โหลด history พร้อมข้อมูลใหม่
            history = checkpoint.get('history', {'train_loss': [], 'val_acc': [], 'val_loss': [], 'epoch_times': []})
            total_training_time_overall = checkpoint.get('total_training_time_overall', 0) # โหลดเวลา training สะสม

            logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch + 1} with best_acc {best_acc:.4f}. Total training time so far: {total_training_time_overall:.2f} seconds.")
            return start_epoch, best_acc, patience_counter, history, total_training_time_overall
        except Exception as e:
            logger.warning(f"Error loading checkpoint {filepath}: {e}. Starting training from scratch.")
            return 0, 0, 0, {'train_loss': [], 'val_acc': [], 'val_loss': [], 'epoch_times': []}, 0

# ================== GLOBAL PAUSE/STOP HANDLER ==================
def handle_signals(model, optimizer, scheduler, epoch, current_batch_idx, best_acc, patience_counter, history, model_name_actual, checkpoint_filename_base, model_checkpoint_dir, device, total_training_time_overall_from_checkpoint):
    if os.path.exists(PAUSE_SIGNAL_FILE):
        logger.info(f"PAUSE_SIGNAL detected. Pausing operation for {model_name_actual} at Epoch {epoch+1}, Batch {current_batch_idx+1}.")
        
        original_device = next(model.parameters()).device 
        model.to(torch.device('cpu')) 
        
        save_checkpoint({
            'epoch': epoch, 
            'batch_idx': current_batch_idx, 
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(), 
            'scheduler': scheduler.state_dict(),
            'best_acc': best_acc,
            'patience_counter': patience_counter,
            'history': history,
            'total_training_time_overall': total_training_time_overall_from_checkpoint, # บันทึกเวลา training สะสม
        }, filename=checkpoint_filename_base, checkpoint_dir=model_checkpoint_dir)
        logger.info(f"State saved. Waiting for PAUSE_SIGNAL.txt to be removed...")
        
        while os.path.exists(PAUSE_SIGNAL_FILE):
            time.sleep(5) 
        
        logger.info(f"PAUSE_SIGNAL cleared. Resuming operation for {model_name_actual}.")
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(original_device)

    if os.path.exists(STOP_SIGNAL_FILE):
        logger.info(f"STOP_SIGNAL detected. Terminating training for {model_name_actual}.")
        os.remove(STOP_SIGNAL_FILE) 
        original_device = next(model.parameters()).device 
        model.to(torch.device('cpu')) 
        save_checkpoint({ 
            'epoch': epoch, 
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_acc': best_acc,
            'patience_counter': patience_counter,
            'history': history,
            'total_training_time_overall': total_training_time_overall_from_checkpoint, # บันทึกเวลา training สะสม
        }, filename=checkpoint_filename_base, checkpoint_dir=model_checkpoint_dir)
        logger.info(f"Training terminated by STOP_SIGNAL for {model_name_actual}. Final state saved.")
        return True # ส่งสัญญาณว่าควรหยุดการฝึก

    return False # ไม่มีสัญญาณหยุดหรือหยุดชั่วคราว

# ================== TRAIN (ปรับปรุงการเรียกใช้ handle_signals) ==================
def train_model(model, train_loader, val_loader, device, model_run_output_dir, model_name_actual, epochs, lr, patience, scheduler_patience, checkpoint_filename_base, force_start_from_scratch=False):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=scheduler_patience, factor=0.5) 
    
    model_base_name = model_name_actual.lower().replace("-", "_") 
    
    model_checkpoint_dir = os.path.join(CHECKPOINT_DIR, model_base_name)
    os.makedirs(model_checkpoint_dir, exist_ok=True)

    # เพิ่ม total_training_time_overall_from_checkpoint เข้ามาใน load_checkpoint
    start_epoch, best_acc, patience_counter, history, total_training_time_overall_from_checkpoint = load_checkpoint(
        model, optimizer, scheduler, filename=checkpoint_filename_base, 
        checkpoint_dir=model_checkpoint_dir, force_start_from_scratch=force_start_from_scratch
    )
    
    model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    final_best_epoch_overall = 0 
    if history['val_acc']:
        if best_acc in history['val_acc']:
            initial_best_epoch_idx = np.where(np.array(history['val_acc']) == best_acc)[0][-1]
            final_best_epoch_overall = initial_best_epoch_idx + 1 
        else:
            final_best_epoch_overall = start_epoch 

    best_model_saved_in_current_run = False
    start_time_current_run = time.time() # เริ่มจับเวลาสำหรับรันปัจจุบัน

    for epoch in range(start_epoch, epochs): 
        epoch_start_time = time.time() # เริ่มจับเวลาสำหรับ epoch

        # ตรวจสอบสัญญาณ PAUSE/STOP ก่อนเริ่ม Epoch
        if handle_signals(model, optimizer, scheduler, epoch, 0, best_acc, patience_counter, history, model_name_actual, checkpoint_filename_base, model_checkpoint_dir, device, total_training_time_overall_from_checkpoint):
            train_time_taken_current_run = time.time() - start_time_current_run
            total_training_time_overall = total_training_time_overall_from_checkpoint + train_time_taken_current_run
            
            model_save_dir = os.path.join(model_run_output_dir, 'models')
            os.makedirs(model_save_dir, exist_ok=True)
            best_model_current_filepath = os.path.join(model_save_dir, f"{model_base_name}_best.pth")
            if not best_model_saved_in_current_run and not os.path.exists(best_model_current_filepath):
                potential_best_paths = glob.glob(os.path.join(RESULTS_BASE_DIR, '*', model_base_name, 'models', f"{model_base_name}_best.pth"))
                potential_best_paths.sort(key=os.path.getmtime, reverse=True)
                if potential_best_paths:
                    shutil.copy2(potential_best_paths[0], best_model_current_filepath)
                    logger.info(f"Copied best model from previous run ({potential_best_paths[0]}) to current run folder: {best_model_current_filepath}")
            
            return best_acc, history, total_training_time_overall, optimizer, scheduler 
            
        model.train()
        running_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} (Train)", unit="batch") as pbar:
            for batch_idx, (images, labels) in enumerate(pbar): 
                # ตรวจสอบสัญญาณ PAUSE/STOP ระหว่าง Batch
                if handle_signals(model, optimizer, scheduler, epoch, batch_idx, best_acc, patience_counter, history, model_name_actual, checkpoint_filename_base, model_checkpoint_dir, device, total_training_time_overall_from_checkpoint):
                    train_time_taken_current_run = time.time() - start_time_current_run
                    total_training_time_overall = total_training_time_overall_from_checkpoint + train_time_taken_current_run
                    
                    model_save_dir = os.path.join(model_run_output_dir, 'models')
                    os.makedirs(model_save_dir, exist_ok=True)
                    best_model_current_filepath = os.path.join(model_save_dir, f"{model_base_name}_best.pth")
                    if not best_model_saved_in_current_run and not os.path.exists(best_model_current_filepath):
                        potential_best_paths = glob.glob(os.path.join(RESULTS_BASE_DIR, '*', model_base_name, 'models', f"{model_base_name}_best.pth"))
                        potential_best_paths.sort(key=os.path.getmtime, reverse=True)
                        if potential_best_paths:
                            shutil.copy2(potential_best_paths[0], best_model_current_filepath)
                            logger.info(f"Copied best model from previous run ({potential_best_paths[0]}) to current run folder: {best_model_current_filepath}")
                    
                    return best_acc, history, total_training_time_overall, optimizer, scheduler 

                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                pbar.set_postfix(loss=running_loss/((batch_idx + 1) * images.size(0)), refresh=True)
            
        avg_train_loss = running_loss / len(train_loader.dataset)
        
        # ตรวจสอบสัญญาณ PAUSE/STOP ก่อน Evaluate
        if handle_signals(model, optimizer, scheduler, epoch, 0, best_acc, patience_counter, history, model_name_actual, checkpoint_filename_base, model_checkpoint_dir, device, total_training_time_overall_from_checkpoint):
            train_time_taken_current_run = time.time() - start_time_current_run
            total_training_time_overall = total_training_time_overall_from_checkpoint + train_time_taken_current_run
            
            model_save_dir = os.path.join(model_run_output_dir, 'models')
            os.makedirs(model_save_dir, exist_ok=True)
            best_model_current_filepath = os.path.join(model_save_dir, f"{model_base_name}_best.pth")
            if not best_model_saved_in_current_run and not os.path.exists(best_model_current_filepath):
                potential_best_paths = glob.glob(os.path.join(RESULTS_BASE_DIR, '*', model_base_name, 'models', f"{model_base_name}_best.pth"))
                potential_best_paths.sort(key=os.path.getmtime, reverse=True)
                if potential_best_paths:
                    shutil.copy2(potential_best_paths[0], best_model_current_filepath)
                    logger.info(f"Copied best model from previous run ({potential_best_paths[0]}) to current run folder: {best_model_current_filepath}")
            
            return best_acc, history, total_training_time_overall, optimizer, scheduler 

        val_acc, avg_val_loss, _ = evaluate(model, val_loader, device, print_report=False, model_name=model_name_actual, 
                                            check_pause_stop_signal=True, 
                                            model_state=(model, optimizer, scheduler, epoch, best_acc, patience_counter, history, model_name_actual, checkpoint_filename_base, model_checkpoint_dir, device, total_training_time_overall_from_checkpoint))
        
        if val_acc is None: 
            train_time_taken_current_run = time.time() - start_time_current_run
            total_training_time_overall = total_training_time_overall_from_checkpoint + train_time_taken_current_run
            return best_acc, history, total_training_time_overall, optimizer, scheduler
            
        history['train_loss'].append(avg_train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(avg_val_loss) # บันทึก val_loss

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        history['epoch_times'].append(epoch_duration) # บันทึกเวลาต่อ epoch
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Epoch Time: {epoch_duration:.2f}s")
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.7f}") 
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            final_best_epoch_overall = epoch + 1 
            model_save_dir = os.path.join(model_run_output_dir, 'models')
            os.makedirs(model_save_dir, exist_ok=True)
            
            best_model_current_filepath = os.path.join(model_save_dir, f"{model_base_name}_best.pth")
            
            if os.path.exists(best_model_current_filepath):
                old_timestamp = time.strftime("%Y%m%d-%H%M%S")
                old_model_new_name = os.path.join(model_save_dir, f"{model_base_name}_best_{old_timestamp}.pth")
                shutil.move(best_model_current_filepath, old_model_new_name)
                logger.info(f"Renamed previous best model to: {old_model_new_name}")
            
            torch.save(model.state_dict(), best_model_current_filepath)
            best_model_saved_in_current_run = True
            
            patience_counter = 0
            logger.info(f"New best model saved to: {best_model_current_filepath} (Val Acc: {best_acc:.4f}).")

        else:
            patience_counter += 1
            
        save_checkpoint({
            'epoch': epoch + 1, 
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_acc': best_acc,
            'patience_counter': patience_counter,
            'history': history, 
            'total_training_time_overall': total_training_time_overall_from_checkpoint + (time.time() - start_time_current_run), # บันทึกเวลา training สะสม
        }, filename=checkpoint_filename_base, checkpoint_dir=model_checkpoint_dir)

        if patience_counter >= patience:
            logger.info(f"Early stopping: Training stopped at epoch {epoch+1} as validation accuracy did not improve for {patience} consecutive epochs.")
            break
            
    end_time_current_run = time.time()
    train_time_taken_current_run = end_time_current_run - start_time_current_run
    total_training_time_overall = total_training_time_overall_from_checkpoint + train_time_taken_current_run
    
    logger.info(f"Training complete. Best Val Acc: {best_acc:.4f} (epoch {final_best_epoch_overall}). Total training time for this model: {total_training_time_overall:.2f} seconds.")

    model_save_dir = os.path.join(model_run_output_dir, 'models')
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_current_filepath = os.path.join(model_save_dir, f"{model_base_name}_best.pth")
    if not best_model_saved_in_current_run and not os.path.exists(best_model_current_filepath):
        potential_best_paths = glob.glob(os.path.join(RESULTS_BASE_DIR, '*', model_base_name, 'models', f"{model_base_name}_best.pth"))
        potential_best_paths.sort(key=os.path.getmtime, reverse=True)
        if potential_best_paths:
            shutil.copy2(potential_best_paths[0], best_model_current_filepath)
            logger.info(f"Copied best model from previous run ({potential_best_paths[0]}) to current run folder: {best_model_current_filepath}")
        else:
            logger.warning(f"Could not find any previous best model for {model_name_actual} to copy.")

    return best_acc, history, total_training_time_overall, optimizer, scheduler

# ================== EVALUATE (ปรับปรุง Progress Bar และเพิ่มการตรวจสอบสัญญาณ) ==================
def evaluate(model, loader, device, print_report=True, output_plot_dir=None, model_name="", check_pause_stop_signal=False, model_state=None):
    model.eval()
    all_preds = []
    all_labels = []
    
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    eval_bar_format = '{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    
    with tqdm(loader, desc="Evaluating...", unit="batch", bar_format=eval_bar_format) as pbar:
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(pbar):
                # ตรวจสอบสัญญาณ PAUSE/STOP ภายใน Evaluate ด้วยงับ
                if check_pause_stop_signal and model_state:
                    model_obj, optimizer_obj, scheduler_obj, epoch_obj, best_acc_obj, patience_counter_obj, history_obj, model_name_actual_obj, checkpoint_filename_base_obj, model_checkpoint_dir_obj, device_obj, total_training_time_overall_from_checkpoint_obj = model_state
                    
                    if handle_signals(model_obj, optimizer_obj, scheduler_obj, epoch_obj, batch_idx, best_acc_obj, patience_counter_obj, history_obj, model_name_actual_obj, checkpoint_filename_base_obj, model_checkpoint_dir_obj, device_obj, total_training_time_overall_from_checkpoint_obj):
                        logger.info(f"Evaluation for {model_name} interrupted by signal.")
                        return None, None, None 

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
        logger.info(f"Overall Accuracy: {acc:.4f}")
        
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8,6))
        plt.imshow(cm, cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(np.arange(len(LABELS)), LABELS, rotation=45)
        plt.yticks(np.arange(len(LABELS)), LABELS)
        plt.colorbar()
        plt.tight_layout()

        if output_plot_dir:
            os.makedirs(output_plot_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            figure_filename = f'confusion_matrix_{model_name.lower().replace("-", "_")}_{timestamp}.png'
            plt.savefig(os.path.join(output_plot_dir, figure_filename))
            logger.info(f"Confusion Matrix saved to: {os.path.join(output_plot_dir, figure_filename)}")
        else:
            logger.warning("No output_plot_dir provided for Confusion Matrix. Plot will not be saved.")
        plt.close()
    return acc, avg_loss, report

# ================== MAIN ==================
if __name__ == '__main__':
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Please check your PyTorch and NVIDIA Driver installation.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running on Device: {device}")

    current_run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    current_run_results_dir = os.path.join(RESULTS_BASE_DIR, current_run_timestamp)
    os.makedirs(current_run_results_dir, exist_ok=True)
    logger.info(f"Results for this run will be saved in: {current_run_results_dir}")

    results = [] # สำหรับ overall summary CSV

    print("\n----- Training Options -----")
    print("1. Start new training (delete ALL old results and checkpoints)")
    print("2. Start new training (archive ALL old results and checkpoints with timestamp)")
    print("3. Resume training from last checkpoint (per model)")
    print("4. Exit")

    choice = input("Please choose an option (1/2/3/4): ")

    force_start_from_scratch_flag = False
    
    if choice == '1':
        if os.path.exists(RESULTS_BASE_DIR):
            shutil.rmtree(RESULTS_BASE_DIR)
            logger.info(f"Removed existing results directory: {RESULTS_BASE_DIR}")
        if os.path.exists(CHECKPOINT_DIR):
            shutil.rmtree(CHECKPOINT_DIR)
            logger.info(f"Removed existing checkpoint directory: {CHECKPOINT_DIR}")
        os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        force_start_from_scratch_flag = True
        logger.info("Option 1 selected: Starting new training from scratch, all old files removed.")

    elif choice == '2':
        timestamp_for_archive = time.strftime("%Y%m%d-%H%M%S")
        if os.path.exists(BASE_OUTPUT_DIR):
            new_archived_base_dir = f"{BASE_OUTPUT_DIR}_archived_{timestamp_for_archive}"
            os.rename(BASE_OUTPUT_DIR, new_archived_base_dir)
            logger.info(f"Archived existing '{BASE_OUTPUT_DIR}' directory to: {new_archived_base_dir}")
        
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        os.makedirs(RESULTS_BASE_DIR, exist_ok=True) 
        os.makedirs(CHECKPOINT_DIR, exist_ok=True) 
        
        force_start_from_scratch_flag = True
        logger.info("Option 2 selected: Starting new training from scratch, all old files archived.")

    elif choice == '3':
        force_start_from_scratch_flag = False
        logger.info("Option 3 selected: Resuming training from last checkpoint (per model).")
    elif choice == '4':
        logger.info("Option 4 selected: Exiting the program. Goodbye!")
        exit()
    else:
        logger.warning("Invalid choice. Defaulting to 'Resume training from last checkpoint'.")
        force_start_from_scratch_flag = False

    for model_name_to_train in MODEL_NAMES:
        model_base_name_lower = model_name_to_train.lower().replace("-", "_") 
        
        model_run_output_dir = os.path.join(current_run_results_dir, model_base_name_lower)
        os.makedirs(model_run_output_dir, exist_ok=True)
        
        checkpoint_filename = f'{model_base_name_lower}_checkpoint.pth.tar'

        model_checkpoint_dir = os.path.join(CHECKPOINT_DIR, model_base_name_lower)
        os.makedirs(model_checkpoint_dir, exist_ok=True) 

        model_hparams = get_model_hyperparameters(model_name_to_train)
        epochs_for_model = model_hparams['epochs']
        lr_for_model = model_hparams['lr']
        patience_for_model = model_hparams['patience']
        scheduler_patience_for_model = model_hparams['scheduler_patience']
        batch_size_for_model = model_hparams['batch_size']

        logger.info(f"\n{'='*50}")
        logger.info(f"===== Starting training for model: {model_name_to_train} =====")
        logger.info(f"Hyperparameters: Epochs={epochs_for_model}, LR={lr_for_model}, Patience={patience_for_model}, Scheduler Patience={scheduler_patience_for_model}, Batch Size={batch_size_for_model}")
        logger.info(f"Results will be saved in: {model_run_output_dir}")
        logger.info(f"{'='*50}")
        
        model = get_model(model_name_to_train, num_classes=len(LABELS))
        
        logger.info(f"Initiating training for {model_name_to_train}...")
        
        train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size_for_model)

        # **สำคัญ:** รับค่าที่ return มาให้ครบ (เพิ่ม total_training_time_overall เข้ามา)
        best_val_acc, history, total_training_time_overall, optimizer, scheduler = train_model(
            model, train_loader, val_loader, device, model_run_output_dir, model_name_to_train,
            epochs=epochs_for_model, 
            lr=lr_for_model, 
            patience=patience_for_model, 
            scheduler_patience=scheduler_patience_for_model,
            checkpoint_filename_base=checkpoint_filename,
            force_start_from_scratch=force_start_from_scratch_flag
        )
        
        # --- บันทึก Training History รายละเอียดต่อ Epoch ---
        training_history_output_dir = os.path.join(model_run_output_dir, 'logs')
        os.makedirs(training_history_output_dir, exist_ok=True)
        
        if history['train_loss'] and history['val_acc'] and history['val_loss'] and history['epoch_times']:
            history_df = pd.DataFrame({
                'Epoch': range(1, len(history['train_loss']) + 1), 
                'Train Loss': history['train_loss'],
                'Val Accuracy': history['val_acc'],
                'Val Loss': history['val_loss'], # เพิ่ม Val Loss
                'Epoch Duration (seconds)': history['epoch_times'] # เพิ่มเวลาต่อ Epoch
            })
            model_timestamp_for_files = time.strftime("%Y%m%d-%H%M%S")
            history_csv_filename = os.path.join(training_history_output_dir, f'{model_base_name_lower}_training_history_{model_timestamp_for_files}.csv')
            history_df.to_csv(history_csv_filename, index=False)
            logger.info(f"Training history for {model_name_to_train} saved to: {history_csv_filename}")
        else:
            logger.warning(f"No complete training history data available for {model_name_to_train} to save.")
        
        logger.info(f"\n--- Evaluation of {model_name_to_train} on Test Set ---")
        
        model_models_dir = os.path.join(model_run_output_dir, 'models')
        best_model_filepath_for_eval = os.path.join(model_models_dir, f"{model_base_name_lower}_best.pth")
        
        test_acc = 0
        test_loss = 0
        test_report = None

        if os.path.exists(best_model_filepath_for_eval):
            logger.info(f"Loading best model from current run folder: {best_model_filepath_for_eval}")
            model.load_state_dict(torch.load(best_model_filepath_for_eval, map_location=device))
            model.to(device) 
            
            confusion_matrix_output_dir = os.path.join(model_run_output_dir, 'plots')
            current_epoch_for_state = len(history['train_loss']) if history['train_loss'] else 0
            
            test_acc, test_loss, test_report = evaluate(model, test_loader, device, print_report=True, 
                                                         output_plot_dir=confusion_matrix_output_dir, 
                                                         model_name=model_name_to_train,
                                                         check_pause_stop_signal=True,
                                                         model_state=(model, optimizer, scheduler, current_epoch_for_state, best_val_acc, patience_for_model, history, model_name_to_train, checkpoint_filename, model_checkpoint_dir, device, total_training_time_overall))
        else:
            logger.warning(f"No best model .pth file found in current run folder: {best_model_filepath_for_eval}.")
            potential_best_paths = glob.glob(os.path.join(RESULTS_BASE_DIR, '*', model_base_name_lower, 'models', f"{model_base_name_lower}_best.pth"))
            potential_best_paths.sort(key=os.path.getmtime, reverse=True)
            
            if potential_best_paths:
                best_model_from_previous_run = potential_best_paths[0]
                logger.info(f"Attempting to load best model from previous run: {best_model_from_previous_run}")
                try:
                    model.load_state_dict(torch.load(best_model_from_previous_run, map_location=device))
                    model.to(device)
                    shutil.copy2(best_model_from_previous_run, best_model_filepath_for_eval)
                    logger.info(f"Copied {os.path.basename(best_model_from_previous_run)} to current run folder for evaluation: {best_model_filepath_for_eval}")
                    
                    confusion_matrix_output_dir = os.path.join(model_run_output_dir, 'plots')
                    current_epoch_for_state = len(history['train_loss']) if history['train_loss'] else 0
                    
                    test_acc, test_loss, test_report = evaluate(model, test_loader, device, print_report=True, 
                                                                 output_plot_dir=confusion_matrix_output_dir, 
                                                                 model_name=model_name_to_train,
                                                                 check_pause_stop_signal=True,
                                                                 model_state=(model, optimizer, scheduler, current_epoch_for_state, best_val_acc, patience_for_model, history, model_name_to_train, checkpoint_filename, model_checkpoint_dir, device, total_training_time_overall))
                except Exception as e:
                    logger.error(f"Error loading best model from {best_model_from_previous_run}: {e}. Cannot perform test evaluation.")
            else:
                logger.error(f"No best model .pth file found at {best_model_filepath_for_eval} or in any previous run. Cannot perform test evaluation.")
        
        # --- บันทึก Test Evaluation Results ของแต่ละโมเดลลง CSV ทันที ---
        if test_report:
            model_test_results_data = {
                'Model': [model_name_to_train],
                'Test Accuracy': [test_acc],
                'Test Loss': [test_loss],
                'Total Training Time (seconds)': [total_training_time_overall], # ใช้เวลาสะสมแล้ว
                'Best Validation Accuracy': [best_val_acc],
                'Test Precision (Weighted Avg)': [test_report['weighted avg']['precision']],
                'Test Recall (Weighted Avg)': [test_report['weighted avg']['recall']],
                'Test F1-Score (Weighted Avg)': [test_report['weighted avg']['f1-score']]
            }

            for label_name in LABELS:
                if label_name in test_report:
                    model_test_results_data[f'Test Precision ({label_name})'] = [test_report[label_name]['precision']]
                    model_test_results_data[f'Test Recall ({label_name})'] = [test_report[label_name]['recall']]
                    model_test_results_data[f'Test F1-Score ({label_name})'] = [test_report[label_name]['f1-score']]
            
            model_test_results_df = pd.DataFrame(model_test_results_data)
            
            # บันทึก CSV ของโมเดลปัจจุบันในโฟลเดอร์ของมัน
            model_test_csv_filename = os.path.join(model_run_output_dir, f'test_evaluation_results_{model_base_name_lower}_{current_run_timestamp}.csv')
            model_test_results_df.to_csv(model_test_csv_filename, index=False)
            logger.info(f"Test evaluation results for {model_name_to_train} saved to: {model_test_csv_filename}")
        else:
            logger.warning(f"No test report generated for {model_name_to_train}. Skipping saving test evaluation CSV for this model.")

        # รวบรวมผลสำหรับ overall summary CSV (model_comparison_results)
        model_summary_results = {
            'Model': model_name_to_train,
            'Best Validation Accuracy': best_val_acc,
            'Test Accuracy': test_acc,
            'Total Training Time (seconds)': total_training_time_overall, # ใช้เวลาสะสม
        }
        if test_report:
            model_summary_results['Test Loss'] = test_loss
            model_summary_results['Test Precision (Weighted Avg)'] = test_report['weighted avg']['precision']
            model_summary_results['Test Recall (Weighted Avg)'] = test_report['weighted avg']['recall']
            model_summary_results['Test F1-Score (Weighted Avg)'] = test_report['weighted avg']['f1-score']
        results.append(model_summary_results)


    results_df = pd.DataFrame(results)
    csv_filename = os.path.join(current_run_results_dir, f'model_comparison_results_{current_run_timestamp}.csv')
    results_df.to_csv(csv_filename, index=False)
    logger.info(f"\n--- All model comparisons completed! ---")
    logger.info(f"Summary results saved to: {csv_filename}")

    logger.info("\nSummary of Model Comparisons:")
    logger.info(results_df.T.to_string())