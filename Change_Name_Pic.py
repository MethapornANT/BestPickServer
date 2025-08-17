import os

# กำหนด Path ของโฟลเดอร์ที่คุณต้องการเปลี่ยนชื่อไฟล์
# ให้เปลี่ยน "path/to/your/folder" เป็น Path จริงๆ ของคุณ
folder_path = "./data3/porn"

# ลิสต์ไฟล์ทั้งหมดในโฟลเดอร์
file_list = os.listdir(folder_path)

# วนลูปเพื่อดูไฟล์แต่ละไฟล์
for filename in file_list:
    # ตรวจสอบว่าชื่อไฟล์มี "_1_" อยู่หรือไม่
    if "_6_" in filename:
        # สร้างชื่อไฟล์ใหม่ โดยเปลี่ยน "_1_" เป็น "_10_"
        new_filename = filename.replace("_6_", "_15_")
        
        # กำหนด Path เต็มของไฟล์ปัจจุบันและไฟล์ใหม่
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)
        
        # เปลี่ยนชื่อไฟล์
        os.rename(old_file, new_file)
        
        print(f"เปลี่ยนชื่อไฟล์ {filename} เป็น {new_filename} เรียบร้อยแล้ว")

print("การเปลี่ยนชื่อไฟล์ทั้งหมดเสร็จสมบูรณ์!")