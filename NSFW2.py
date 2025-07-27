import os
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn.functional as F
import pandas as pd # สำหรับจัดการข้อมูลและบันทึกเป็น CSV/Excel
from datetime import datetime # สำหรับตั้งชื่อไฟล์ผลลัพธ์

# --- การตั้งค่าและโหลดโมเดล ---
# กำหนดโฟลเดอร์สำหรับเก็บไฟล์รูปภาพที่จะทำนาย
UPLOAD_FOLDER = './data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # สร้างโฟลเดอร์ถ้ายังไม่มี

# กำหนดเส้นทางไปยังไฟล์โมเดลที่เทรนมาแล้ว
MODEL_PATH = 'vit_b_16_best.pth' 

# Mapping id เป็น label (ต้องตรงกับ LABELS ที่ใช้ตอนเทรนโมเดล ViT-B_16_best.pth)
LABELS = ['normal', 'hentai', 'porn', 'sexy', 'anime']
idx2label = {idx: label for idx, label in enumerate(LABELS)}

# กำหนด Device ที่จะใช้ (GPU ถ้ามี, ไม่งั้นก็ CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- โหลดโมเดล ViT-B_16 ที่เทรนมาแล้ว ---
model = None # กำหนดค่าเริ่มต้นเป็น None
print(f"กำลังโหลดโมเดล ViT-B_16 จากไฟล์: {MODEL_PATH}...")
try:
    model = models.vit_b_16(weights=None) 
    model.heads = torch.nn.Sequential(
        torch.nn.Dropout(0.2), 
        torch.nn.Linear(model.heads.head.in_features, len(LABELS))
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() 
    model.to(device) 
    print(f"โหลดโมเดล {MODEL_PATH} เสร็จสมบูรณ์แล้วงับ! โมเดลทำงานอยู่บน: {device}")

except FileNotFoundError:
    print(f"❌ Error: ไม่พบไฟล์โมเดล '{MODEL_PATH}' โปรดตรวจสอบว่าไฟล์อยู่ในตำแหน่งที่ถูกต้องงับ")
    exit() # ถ้าโมเดลหลักโหลดไม่ได้ก็จบโปรแกรม
except Exception as e:
    print(f"💔 เกิดข้อผิดพลาดในการโหลดหรือเตรียมโมเดลจาก '{MODEL_PATH}': {e}")
    print("โปรดตรวจสอบว่าโมเดล ViT-B_16_best.pth เข้ากันได้กับโครงสร้างที่สร้างขึ้นงับ")
    exit()

# กำหนด transforms ที่ใช้ในการประเมินผล (ต้องเหมือนตอนเทรน)
processor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- ฟังก์ชันสำหรับทำนาย NSFW ---
def nude_predict_image(image_path):
    """
    ฟังก์ชันสำหรับทำนายภาพว่าเป็น NSFW หรือไม่ โดยใช้โมเดล vit_b_16_best.pth
    พร้อมแสดงผลเปอร์เซ็นต์ของแต่ละหมวดหมู่
    และระบุว่าภาพเป็น NSFW ตามเกณฑ์ Hentai > 20% หรือ Pornography > 20%

    Args:
        image_path (str): เส้นทางไปยังไฟล์รูปภาพ

    Returns:
        dict: Dictionary ที่แสดงเปอร์เซ็นต์ของแต่ละหมวดหมู่, is_nsfw,
              หรือข้อความ error หากเกิดปัญหา
    """
    if not os.path.exists(image_path):
        print(f"❌ Error: ไม่พบไฟล์รูปภาพที่ {image_path} งับ")
        return {"error": "ไม่พบไฟล์รูปภาพ กรุณาตรวจสอบเส้นทางอีกครั้งงับ"}

    try:
        image = Image.open(image_path).convert("RGB")
        
        inputs = processor(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            logits = outputs
            probs = F.softmax(logits, dim=1).squeeze().tolist()

        if len(probs) != len(LABELS):
            # print(f"⚠️ คำเตือน: จำนวน output จากโมเดล ({len(probs)}) ไม่ตรงกับจำนวน LABELS ที่กำหนด ({len(LABELS)}) งับ")
            all_scores = {idx2label.get(i, f"Unknown_{i}"): round(probs[i]*100, 2) for i in range(len(probs))}
        else:
            all_scores = {idx2label[i]: round(probs[i]*100, 2) for i in range(len(probs))}

        hentai_score = all_scores.get("hentai", 0.0)
        porn_score = all_scores.get("porn", 0.0)
        
        is_nsfw = hentai_score > 20 or porn_score > 20
        
        print(f"\n✨ ผลการตรวจสอบ NSFW สำหรับ: {os.path.basename(image_path)} ✨")
        for label, score in all_scores.items():
            print(f"  - {label.capitalize()}: {score:.2f}%") 
        
        print(f"--- สรุป: ภาพนี้ {'🔴 เป็น NSFW' if is_nsfw else '🟢 ไม่เป็น NSFW'} (Hentai > 20% หรือ Pornography > 20%) ---")
        
        return {"scores": all_scores, "is_nsfw": is_nsfw}
        
    except Exception as e:
        print(f"💔 เกิดข้อผิดพลาดในการตรวจสอบ NSFW สำหรับ {image_path}: {e}")
        return {"error": f"ไม่สามารถตรวจสอบภาพได้ กรุณาลองใหม่อีกครั้ง: {e}"}

# --- ฟังก์ชันสำหรับเลือกโหมดการทำงาน ---
def get_image_files_in_folder(folder_path):
    """ดึงรายชื่อไฟล์รูปภาพทั้งหมดในโฟลเดอร์ที่กำหนด"""
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(image_extensions)]

def main_menu():
    """แสดงเมนูหลักและรับตัวเลือกจากผู้ใช้"""
    
    # กำหนด header สำหรับ CSV (ต้องตรงกับ LABELS)
    # เพิ่ม 'Status' และ 'is_NSFW' ด้วย
    csv_headers = ['Image_Name', 'Status'] 
    for label in LABELS:
        csv_headers.append(f'{label.capitalize()}_%')
    csv_headers.append('is_NSFW')

    while True:
        print("\n--- เลือกโหมดการทำงาน (Custom Model) งับ ---")
        print("1. ทำนายรูปภาพทั้งหมดในโฟลเดอร์ (./data) และบันทึกผล")
        print("2. เลือกรูปภาพเดียวเพื่อทำนาย (ไม่บันทึกผล)")
        print("3. ออกจากการทำงาน")
        
        choice = input("กรุณาป้อนตัวเลือก (1, 2, หรือ 3): ")

        if choice == '1':
            print(f"\nกำลังทำนายรูปภาพทั้งหมดในโฟลเดอร์: {UPLOAD_FOLDER} งับ...")
            image_paths = get_image_files_in_folder(UPLOAD_FOLDER)
            
            if not image_paths:
                print(f"❌ ไม่พบรูปภาพใดๆ ในโฟลเดอร์ '{UPLOAD_FOLDER}' งับ")
                print("กรุณานำไฟล์รูปภาพไปใส่ไว้ในโฟลเดอร์นั้นก่อนนะงับ")
                continue

            # กำหนดชื่อไฟล์ CSV โดยใช้ Timestamp เพื่อไม่ให้ซ้ำกัน
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"custom_nsfw_results_{timestamp}.csv"
            
            # เก็บผลลัพธ์สำหรับการบันทึก
            batch_results = [] 
            
            for img_path in image_paths:
                result = nude_predict_image(img_path)
                
                row_data = {'Image_Name': os.path.basename(img_path)}
                if "error" in result:
                    row_data['Status'] = 'Error'
                    for label in LABELS:
                        row_data[f'{label.capitalize()}_%'] = 'N/A'
                    row_data['is_NSFW'] = 'N/A'
                    print(f"❌ เกิดข้อผิดพลาดสำหรับ {os.path.basename(img_path)}: {result['error']}")
                else:
                    row_data['Status'] = 'OK'
                    for label_key, score_value in result['scores'].items():
                        row_data[f'{label_key.capitalize()}_%'] = score_value # ชื่อคอลัมน์จะตรงกับ LABELS
                    row_data['is_NSFW'] = result['is_nsfw']
                
                batch_results.append(row_data)
            
            # บันทึกผลลัพธ์ลง CSV/Excel
            if batch_results:
                df_new = pd.DataFrame(batch_results)
                # บันทึกต่อท้ายไฟล์เดิม (ถ้ามี) หรือสร้างใหม่ ถ้ายังไม่มี
                # เป็ดน้อยเปลี่ยนมาสร้างไฟล์ใหม่เสมอ เพื่อให้ข้อมูลแต่ละ batch แยกกันชัดเจน
                df_new.to_csv(output_filename, index=False, encoding='utf-8-sig')
                print(f"\n🎉 บันทึกผลลัพธ์การทำนายทั้งหมดลงในไฟล์: {output_filename} แล้วงับ!")
            else:
                print("\n⚠️ ไม่มีผลลัพธ์ให้บันทึก (อาจเกิดข้อผิดพลาดในการประมวลผลทุกภาพ) งับ")

            print("\n--- การทำนายรูปภาพทั้งหมดในโฟลเดอร์เสร็จสิ้นแล้วงับ ---")

        elif choice == '2':
            real_image_to_test = input("กรุณาป้อน Path ของรูปภาพที่คุณต้องการทำนาย (เช่น data/image.jpg หรือ D:/images/my_pic.png): ")
            if os.path.exists(real_image_to_test):
                nude_predict_image(real_image_to_test) # โหมดนี้ไม่บันทึกลง CSV
            else:
                print(f"❌ ไม่พบไฟล์รูปภาพที่ '{real_image_to_test}' งับ")
                print("โปรดตรวจสอบ Path ของรูปภาพที่คุณป้อนอีกครั้งนะงับ")

        elif choice == '3':
            print("ขอบคุณที่ใช้งานงับ! เป็ดน้อยไปแล้วนะ บ๊ายบาย! 👋")
            break
        else:
            print("⚠️ ตัวเลือกไม่ถูกต้องงับ! กรุณาป้อน 1, 2, หรือ 3 นะงับ")

# --- เริ่มการทำงานเมื่อรันสคริปต์ ---
if __name__ == "__main__":
    main_menu()