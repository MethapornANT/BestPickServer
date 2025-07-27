import os
import torch
from PIL import Image
from transformers import SiglipForImageClassification, AutoImageProcessor # สำหรับโมเดล Hugging Face
import torch.nn.functional as F
import pandas as pd # สำหรับจัดการข้อมูลและบันทึกเป็น CSV/Excel
from datetime import datetime # สำหรับตั้งชื่อไฟล์ผลลัพธ์

# --- การตั้งค่าและโหลดโมเดล Hugging Face ---
# กำหนดโฟลเดอร์สำหรับเก็บไฟล์รูปภาพที่จะทำนาย
UPLOAD_FOLDER = './TestPIC' # ใช้โฟลเดอร์เดียวกันกับโค้ดของคุณ
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # สร้างโฟลเดอร์ถ้ายังไม่มี

# กำหนดชื่อโมเดล Hugging Face
MODEL_NAME_HF = "strangerguardhf/nsfw_image_detection"

# LABELS ที่ต้องการแสดงผลและบันทึก (จัดเรียงตามลำดับเดียวกับ LABELS ของคุณ)
# เป็ดน้อยจะพยายาม map label ของ HF ให้เข้ากับอันนี้นะงับ
TARGET_LABELS = ['normal', 'hentai', 'porn', 'sexy', 'anime']

# กำหนด Device ที่จะใช้ (GPU ถ้ามี, ไม่งั้นก็ CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- โหลดโมเดล Hugging Face (Siglip) ---
model_hf = None
processor_hf = None
print(f"กำลังโหลดโมเดลและ processor จาก Hugging Face: {MODEL_NAME_HF}...")
try:
    model_hf = SiglipForImageClassification.from_pretrained(MODEL_NAME_HF)
    processor_hf = AutoImageProcessor.from_pretrained(MODEL_NAME_HF)
    model_hf.eval()
    model_hf.to(device)
    print(f"โหลดโมเดล '{MODEL_NAME_HF}' เสร็จสมบูรณ์แล้วงับ! โมเดลทำงานอยู่บน: {device}")
except Exception as e:
    print(f"💔 เกิดข้อผิดพลาดในการโหลดโมเดล Hugging Face '{MODEL_NAME_HF}': {e}")
    print("โปรดตรวจสอบการเชื่อมต่ออินเทอร์เน็ตงับ")
    model_hf = None
    processor_hf = None
    exit() # ถ้าโหลดไม่ได้ ก็ไม่สามารถทำงานต่อได้

# --- ฟังก์ชันสำหรับทำนายภาพด้วยโมเดล Hugging Face (ปรับปรุง) ---
def predict_with_hf_model_formatted(image_path):
    """
    ฟังก์ชันสำหรับทำนายภาพด้วยโมเดล Hugging Face
    และจัดรูปแบบผลลัพธ์ให้เป็นไปตาม TARGET_LABELS
    """
    if model_hf is None or processor_hf is None:
        return {"error": "โมเดล Hugging Face โหลดไม่สำเร็จงับ"}
    
    if not os.path.exists(image_path):
        return {"error": f"ไม่พบไฟล์รูปภาพที่ {image_path} งับ"}

    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor_hf(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model_hf(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

        # สร้าง dictionary ผลลัพธ์เบื้องต้นจากโมเดล HF ดั้งเดิม
        original_hf_scores = {model_hf.config.id2label[i]: round(probs[i]*100, 2) for i in range(len(probs))}
        
        # จัดรูปแบบผลลัพธ์ให้ตรงกับ TARGET_LABELS ของคุณ
        formatted_scores = {label.capitalize(): 0.0 for label in TARGET_LABELS} # เริ่มต้นด้วย 0%
        
        # Mapping label จาก HF ไปยัง TARGET_LABELS ของคุณ
        # เป็ดน้อยจะใช้ชื่อ label ของคุณเป็นหลัก และพยายาม map ให้เข้ากัน
        # เช่น 'Normal' -> 'normal', 'Hentai' -> 'hentai', 'Pornography' -> 'porn'
        # 'Enticing or Sensual' อาจจะ map ไป 'sexy'
        # 'Anime Picture' -> 'anime'
        
        # Map แบบ hardcode เพื่อความแน่นอน
        if 'Normal' in original_hf_scores:
            formatted_scores['Normal'] = original_hf_scores['Normal']
        if 'Hentai' in original_hf_scores:
            formatted_scores['Hentai'] = original_hf_scores['Hentai']
        if 'Pornography' in original_hf_scores:
            formatted_scores['Porn'] = original_hf_scores['Pornography'] # เปลี่ยนชื่อเป็น Porn
        if 'Enticing or Sensual' in original_hf_scores:
            formatted_scores['Sexy'] = original_hf_scores['Enticing or Sensual'] # เปลี่ยนชื่อเป็น Sexy
        if 'Anime Picture' in original_hf_scores:
            formatted_scores['Anime'] = original_hf_scores['Anime Picture']


        # ดึงคะแนน Hentai และ Pornography โดยใช้ .get() เพื่อป้องกัน KeyError
        hentai_score = formatted_scores.get("Hentai", 0.0)
        porn_score = formatted_scores.get("Porn", 0.0) # ใช้ 'Porn' ที่เรา map มาแล้ว
        
        # กำหนดเกณฑ์ NSFW
        is_nsfw = hentai_score > 20 or porn_score > 20

        # แสดงผลลัพธ์ใน Terminal
        print(f"\n✨ ผลการตรวจสอบ NSFW สำหรับ: {os.path.basename(image_path)} (จาก Hugging Face Model) ✨")
        for label, score in formatted_scores.items():
            print(f"  - {label}: {score:.2f}%")
        
        print(f"--- สรุป: ภาพนี้ {'🔴 เป็น NSFW' if is_nsfw else '🟢 ไม่เป็น NSFW'} (Hentai > 20% หรือ Pornography > 20%) ---")
        
        return {
            "scores": formatted_scores, 
            "is_nsfw": is_nsfw, 
            "hentai": hentai_score, 
            "porn": porn_score
        }
    except Exception as e:
        print(f"💔 เกิดข้อผิดพลาดในการตรวจสอบ NSFW สำหรับ {image_path}: {e}")
        return {"error": f"ไม่สามารถตรวจสอบภาพได้ กรุณาลองใหม่อีกครั้ง: {e}"}

# --- ฟังก์ชันสำหรับเลือกโหมดการทำงาน ---
def get_image_files_in_folder(folder_path):
    """ดึงรายชื่อไฟล์รูปภาพทั้งหมดในโฟลเดอร์ที่กำหนด"""
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(image_extensions)]

def main_menu_hf():
    """แสดงเมนูหลักและรับตัวเลือกจากผู้ใช้สำหรับโมเดล HF"""
    # กำหนดชื่อไฟล์ CSV โดยใช้ Timestamp เพื่อไม่ให้ซ้ำกัน
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"hf_nsfw_results_{timestamp}.csv"
    
    # กำหนด header สำหรับ CSV (ต้องตรงกับ TARGET_LABELS)
    csv_headers = ['Image_Name', 'Status', 
                   'Normal_%', 'Hentai_%', 'Porn_%', 'Sexy_%', 'Anime_%', 
                   'is_NSFW']

    # สร้างไฟล์ CSV เปล่าๆ พร้อม header หากยังไม่มี
    if not os.path.exists(output_filename):
        with open(output_filename, 'w', encoding='utf-8-sig', newline='') as f:
            writer = pd.DataFrame(columns=csv_headers).to_csv(f, index=False, encoding='utf-8-sig')


    while True:
        print("\n--- เลือกโหมดการทำงาน (Hugging Face Model) งับ ---")
        print("1. ทำนายรูปภาพทั้งหมดในโฟลเดอร์ (./TestPIC) และบันทึกผล")
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

            # เก็บผลลัพธ์สำหรับการบันทึก
            batch_results = [] 
            
            for img_path in image_paths:
                result = predict_with_hf_model_formatted(img_path)
                
                row_data = {'Image_Name': os.path.basename(img_path)}
                if "error" in result:
                    row_data['Status'] = 'Error'
                    for label in TARGET_LABELS:
                        row_data[f'{label.capitalize()}_%'] = 'N/A'
                    row_data['is_NSFW'] = 'N/A'
                    print(f"❌ เกิดข้อผิดพลาดสำหรับ {os.path.basename(img_path)}: {result['error']}")
                else:
                    row_data['Status'] = 'OK'
                    for label_key, score_value in result['scores'].items():
                        row_data[f'{label_key}_%'] = score_value # ชื่อคอลัมน์จะตรงกับ TARGET_LABELS
                    row_data['is_NSFW'] = result['is_nsfw']
                
                batch_results.append(row_data)
            
            # บันทึกผลลัพธ์ลง CSV/Excel
            if batch_results:
                df_new = pd.DataFrame(batch_results)
                # บันทึกต่อท้ายไฟล์เดิม
                df_new.to_csv(output_filename, mode='a', header=False, index=False, encoding='utf-8-sig')
                print(f"\n🎉 บันทึกผลลัพธ์การทำนายทั้งหมดลงในไฟล์: {output_filename} แล้วงับ!")
            else:
                print("\n⚠️ ไม่มีผลลัพธ์ให้บันทึก (อาจเกิดข้อผิดพลาดในการประมวลผลทุกภาพ) งับ")

            print("\n--- การทำนายรูปภาพทั้งหมดในโฟลเดอร์เสร็จสิ้นแล้วงับ ---")

        elif choice == '2':
            real_image_to_test = input("กรุณาป้อน Path ของรูปภาพที่คุณต้องการทำนาย (เช่น TestPIC/image.jpg หรือ D:/images/my_pic.png): ")
            if os.path.exists(real_image_to_test):
                predict_with_hf_model_formatted(real_image_to_test) # โหมดนี้ไม่บันทึกลง CSV
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
    main_menu_hf()