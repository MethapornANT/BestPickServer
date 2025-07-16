import os
from pathlib import Path
import hashlib

def get_file_info(folder_path):
    """ดึงข้อมูลไฟล์ในโฟลเดอร์"""
    if not os.path.exists(folder_path):
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    files_info = []
    
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            file_ext = Path(file).suffix.lower()
            if file_ext in image_extensions:
                try:
                    file_size = os.path.getsize(file_path)
                    files_info.append({
                        'name': file,
                        'size': file_size,
                        'path': file_path
                    })
                except:
                    pass
    
    return files_info

def check_duplicates_and_corrupted():
    """เช็คไฟล์ซ้ำและไฟล์เสีย"""
    picdata_path = "PicData"
    
    if not os.path.exists(picdata_path):
        print("❌ ไม่พบโฟลเดอร์ PicData")
        return
    
    print("=" * 60)
    print("🔍 ตรวจสอบไฟล์ภาพใน PicData")
    print("=" * 60)
    
    categories = ["anime", "hentai", "normal", "porn", "sexy"]
    total_files = 0
    total_duplicates = 0
    total_corrupted = 0
    
    for category in categories:
        category_path = os.path.join(picdata_path, category)
        files_info = get_file_info(category_path)
        
        if not files_info:
            print(f"📁 {category:10} | ไม่มีไฟล์")
            continue
        
        # เช็คไฟล์เสีย (ขนาด 0 หรือเล็กเกินไป)
        corrupted_files = [f for f in files_info if f['size'] < 1000]  # น้อยกว่า 1KB
        corrupted_count = len(corrupted_files)
        
        # เช็คไฟล์ซ้ำ (ชื่อซ้ำ)
        file_names = [f['name'] for f in files_info]
        unique_names = set(file_names)
        duplicate_count = len(file_names) - len(unique_names)
        
        print(f"📁 {category:10} | {len(files_info):6} รูป | ซ้ำ {duplicate_count:3} | เสีย {corrupted_count:3}")
        
        if corrupted_files:
            print(f"    ⚠️  ไฟล์เสีย: {[f['name'] for f in corrupted_files[:5]]}")
            if len(corrupted_files) > 5:
                print(f"    ... และอีก {len(corrupted_files) - 5} ไฟล์")
        
        total_files += len(files_info)
        total_duplicates += duplicate_count
        total_corrupted += corrupted_count
    
    print("-" * 60)
    print(f"📊 สรุป: {total_files} ไฟล์ | ซ้ำ {total_duplicates} | เสีย {total_corrupted}")
    print("=" * 60)

if __name__ == "__main__":
    check_duplicates_and_corrupted() 