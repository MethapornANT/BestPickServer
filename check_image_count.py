import os
from pathlib import Path

def count_images_in_folder(folder_path):
    """นับจำนวนไฟล์ภาพในโฟลเดอร์"""
    if not os.path.exists(folder_path):
        return 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    count = 0
    
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            file_ext = Path(file).suffix.lower()
            if file_ext in image_extensions:
                count += 1
    
    return count

def check_data_folders():
    """เช็คจำนวนภาพในแต่ละ folder ใน data"""
    data_path = "data"
    
    if not os.path.exists(data_path):
        print("❌ ไม่พบโฟลเดอร์ data")
        return
    
    print("=" * 50)
    print("📊 รายงานจำนวนภาพใน data")
    print("=" * 50)
    
    total_images = 0
    categories = ["anime", "hentai", "normal", "porn", "sexy"]
    
    for category in categories:
        category_path = os.path.join(data_path, category)
        count = count_images_in_folder(category_path)
        total_images += count
        
        # แสดงผลแบบตาราง
        print(f"📁 {category:10} | {count:6} รูป")
    
    print("-" * 50)
    print(f"📈 รวมทั้งหมด     | {total_images:6} รูป")
    print("=" * 50)
    
    # แสดงรายละเอียดเพิ่มเติม
    print("\n📋 รายละเอียดเพิ่มเติม:")
    for category in categories:
        category_path = os.path.join(data_path, category)
        if os.path.exists(category_path):
            count = count_images_in_folder(category_path)
            if count > 0:
                print(f"  • {category}: {count} รูป")
            else:
                print(f"  • {category}: ไม่มีรูป")
        else:
            print(f"  • {category}: ไม่มีโฟลเดอร์")

if __name__ == "__main__":
    check_data_folders() 