import os
import shutil
import glob
from sklearn.model_selection import train_test_split
from collections import defaultdict
import logging
import numpy as np

# ================== CONFIG ==================
DATA_DIR = './data2' 
TRAIN_DIR = './nsfw_data3/train'
VAL_DIR = './nsfw_data3/val'
TEST_DIR = './nsfw_data3/test'

LABELS = ['normal', 'hentai', 'porn', 'sexy', 'anime'] # ตรวจสอบให้แน่ใจว่าตรงกับชื่อโฟลเดอร์ใน DATA_DIR

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# ================== LOGGING SETUP ==================
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def create_dirs_if_not_exists():
    """สร้างโครงสร้างโฟลเดอร์สำหรับ Train, Val, Test และโฟลเดอร์ย่อยของแต่ละ Label"""
    for base_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(base_dir, exist_ok=True)
        for label in LABELS:
            os.makedirs(os.path.join(base_dir, label), exist_ok=True)
    logger.info("Created necessary directories for train, val, test splits.")

def clean_target_directories():
    """ลบข้อมูลเก่าในโฟลเดอร์ Train, Val, Test ก่อนที่จะ Copy ใหม่"""
    for base_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
            logger.info(f"Cleaned existing directory: {base_dir}")
    create_dirs_if_not_exists() # สร้างใหม่หลังจากลบ

def get_image_paths_by_label():
    """รวบรวมพาร์ทรูปภาพทั้งหมดและ Label ของแต่ละรูป"""
    all_image_paths = []
    all_labels = []
    
    logger.info(f"Collecting image paths from {DATA_DIR}...")
    for label in LABELS:
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_dir):
            logger.warning(f"Label directory '{label_dir}' not found. Skipping this label.")
            continue
        
        # รวบรวมไฟล์รูปภาพทุกนามสกุลที่รองรับ
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']: # เพิ่มนามสกุลที่รองรับ
            for img_path in glob.glob(os.path.join(label_dir, ext)):
                all_image_paths.append(img_path)
                all_labels.append(label)
                
    logger.info(f"Found {len(all_image_paths)} images in total across all labels.")
    return np.array(all_image_paths), np.array(all_labels)

def copy_files(file_paths, target_base_dir):
    """Copy ไฟล์ไปยังโฟลเดอร์ปลายทางที่ถูกต้อง"""
    for img_path in file_paths:
        try:
            # ดึงชื่อ Label จากพาร์ทไฟล์ (parent directory name)
            label = os.path.basename(os.path.dirname(img_path))
            dest_dir = os.path.join(target_base_dir, label)
            shutil.copy2(img_path, dest_dir)
        except Exception as e:
            logger.error(f"Failed to copy {img_path} to {dest_dir}: {e}")

def main():
    logger.info("Starting data splitting process...")
    
    # 1. ทำความสะอาดโฟลเดอร์ปลายทางเดิม
    clean_target_directories()
    
    # 2. รวบรวมพาร์ทรูปภาพและ Label ทั้งหมด
    all_image_paths, all_labels = get_image_paths_by_label()
    
    if len(all_image_paths) == 0:
        logger.error("No images found in DATA_DIR. Please check your data path and folder structure.")
        return

    # 3. แบ่งข้อมูลเป็น Train, Temp (Val+Test) โดยใช้ Stratified Split
    # เพื่อให้แน่ใจว่าสัดส่วนของแต่ละคลาสถูกรักษาไว้
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_image_paths, all_labels, 
        test_size=(VAL_SPLIT + TEST_SPLIT), 
        random_state=42, # กำหนดค่านี้เพื่อให้ผลลัพธ์คงที่ทุกครั้งที่รัน
        stratify=all_labels
    )
    logger.info(f"Initial split: Train={len(X_train)} images, Temp (Val+Test)={len(X_temp)} images.")

    # 4. แบ่ง Temp (Val+Test) ออกเป็น Validation และ Test โดยใช้ Stratified Split อีกครั้ง
    # คำนวณ test_size ใหม่เทียบกับ X_temp
    # เช่น ถ้า temp_size = 0.2, val_size = 0.1, test_size = 0.1
    # test_size_relative_to_temp = TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT) = 0.1 / 0.2 = 0.5
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=(TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)), 
        random_state=42,
        stratify=y_temp
    )
    logger.info(f"Final split: Val={len(X_val)} images, Test={len(X_test)} images.")
    
    # 5. Copy ไฟล์ไปยังโฟลเดอร์ปลายทาง
    logger.info("Copying files to TRAIN_DIR...")
    copy_files(X_train, TRAIN_DIR)
    logger.info("Copying files to VAL_DIR...")
    copy_files(X_val, VAL_DIR)
    logger.info("Copying files to TEST_DIR...")
    copy_files(X_test, TEST_DIR)
    
    logger.info("\nData splitting and copying complete!")
    logger.info(f"Total images: {len(all_image_paths)}")
    logger.info(f"  Train set: {len(X_train)} images ({TRAIN_SPLIT*100:.0f}%)")
    logger.info(f"  Validation set: {len(X_val)} images ({VAL_SPLIT*100:.0f}%)")
    logger.info(f"  Test set: {len(X_test)} images ({TEST_SPLIT*100:.0f}%)")
    
    # ตรวจสอบจำนวนไฟล์ในแต่ละโฟลเดอร์ย่อย (Optional)
    logger.info("\nVerifying counts in target directories:")
    for base_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        logger.info(f"--- {os.path.basename(base_dir).upper()} ---")
        for label in LABELS:
            count = len(glob.glob(os.path.join(base_dir, label, '*.*')))
            logger.info(f"  {label}: {count} images")

if __name__ == '__main__':
    main()