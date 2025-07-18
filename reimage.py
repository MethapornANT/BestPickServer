import os

def convert_all_to_jpg(folder_path="profile"):
    """
    เปลี่ยนนามสกุลของทุกไฟล์ในโฟลเดอร์ที่กำหนดให้เป็น .jpg

    Args:
        folder_path (str): พาธของโฟลเดอร์ที่ต้องการเปลี่ยนนามสกุลไฟล์
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return

    print(f"--- Starting conversion of all files in '{folder_path}' to .jpg ---")

    for filename in os.listdir(folder_path):
        # แยกชื่อไฟล์และนามสกุลปัจจุบัน
        base_name, current_extension = os.path.splitext(filename)

        # ตรวจสอบว่านามสกุลปัจจุบันไม่ใช่ .jpg (โดยไม่คำนึงถึงตัวพิมพ์เล็ก/ใหญ่)
        if current_extension.lower() != '.jpg' and current_extension.lower() != '.jpeg': # .jpeg ก็ถือว่าเป็น jpg เหมือนกัน
            old_filepath = os.path.join(folder_path, filename)
            new_filename = f"{base_name}.jpg"
            new_filepath = os.path.join(folder_path, new_filename)

            # ตรวจสอบว่าไฟล์ปลายทางมีอยู่แล้วหรือไม่ เพื่อป้องกันการทับไฟล์ที่สำคัญ
            if os.path.exists(new_filepath):
                print(f"Warning: '{new_filepath}' already exists. Cannot convert '{filename}'. Skipping.")
                continue

            try:
                os.rename(old_filepath, new_filepath)
                print(f"Converted '{filename}' to '{new_filename}'")
            except OSError as e:
                print(f"Error converting '{filename}': {e}")
        else:
            print(f"'{filename}' is already a .jpg file. No change needed.")

    print("--- Conversion completed ---")

# --- การเรียกใช้งานโค้ด ---
if __name__ == "__main__":
    target_folder = "profile" # กำหนดโฟลเดอร์ที่คุณต้องการให้โค้ดทำงาน

    # สร้างโฟลเดอร์ profile จำลองหากไม่มี (สำหรับทดสอบ)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created '{target_folder}' folder for demonstration.")

    # สร้างไฟล์จำลองในโฟลเดอร์ profile เพื่อทดสอบ
    dummy_files = [
        "image1.png",
        "photo.jpeg",
        "document.txt",
        "video.mp4",
        "already_jpg.jpg",
        "another_image.gif",
        "my_pic.webp" # เพิ่มการทดสอบไฟล์ .webp
    ]
    
    print(f"\n--- Creating dummy files in '{target_folder}' ---")
    for f in dummy_files:
        file_path = os.path.join(target_folder, f)
        if not os.path.exists(file_path):
            open(file_path, "w").close()
            print(f"Created dummy file: {f}")
        else:
            print(f"Dummy file already exists: {f}")
            
    # สร้างสถานการณ์ที่มีชื่อซ้ำกัน แต่คนละนามสกุล
    if not os.path.exists(os.path.join(target_folder, "report.doc")):
        open(os.path.join(target_folder, "report.doc"), "w").close()
        print("Created dummy file: report.doc")
    if not os.path.exists(os.path.join(target_folder, "report.jpg")):
        open(os.path.join(target_folder, "report.jpg"), "w").close()
        print("Created dummy file: report.jpg (to test conflict)")


    print("\n--- Running conversion ---")
    convert_all_to_jpg(target_folder) # รันฟังก์ชัน

    print(f"\n--- Files in '{target_folder}' after conversion ---")
    for file in os.listdir(target_folder):
        print(file)