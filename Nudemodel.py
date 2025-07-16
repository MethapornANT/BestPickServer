import os
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import shutil
from pathlib import Path
import json

class NSFWModel:
    def __init__(self):
        # โหลดโมเดลและ processor สำหรับ NSFW detection
        self.MODEL_NAME = "strangerguardhf/nsfw_image_detection"
        self.model = SiglipForImageClassification.from_pretrained(self.MODEL_NAME)
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_NAME)
        
        # mapping id เป็น label
        self.id2label = {
            "0": "anime",
            "1": "hentai", 
            "2": "normal",
            "3": "porn",
            "4": "sexy"
        }
        
        # สร้าง folder สำหรับเก็บผลลัพธ์
        self.result_folder = "ResultModel"
        self.create_result_folders()
        
    def create_result_folders(self):
        """สร้าง folder สำหรับเก็บผลลัพธ์การจำแนก"""
        if os.path.exists(self.result_folder):
            shutil.rmtree(self.result_folder)
        
        os.makedirs(self.result_folder, exist_ok=True)
        
        # สร้าง subfolder สำหรับแต่ละหมวดหมู่
        for category in self.id2label.values():
            os.makedirs(os.path.join(self.result_folder, category), exist_ok=True)
    
    def predict_image(self, image_path):
        """ทำนายหมวดหมู่ของภาพ"""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
            
            # หาหมวดหมู่ที่มีความน่าจะเป็นสูงสุด
            predicted_class = probs.index(max(probs))
            predicted_category = self.id2label[str(predicted_class)]
            
            # สร้าง dictionary ของผลลัพธ์
            results = {
                "predicted_category": predicted_category,
                "confidence": round(max(probs) * 100, 2),
                "all_probabilities": {self.id2label[str(i)]: round(probs[i] * 100, 2) for i in range(len(probs))}
            }
            
            return predicted_category, results
            
        except Exception as e:
            print(f"Error predicting image {image_path}: {e}")
            return "normal", {"error": str(e)}
    
    def get_all_images_from_data_folder(self):
        """ดึงรูปภาพทั้งหมดจาก folder data และ subfolder"""
        data_folder = "data"
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        images = []
        
        if not os.path.exists(data_folder):
            print(f"Folder {data_folder} ไม่พบ")
            return images
        
        # ใช้ Path เพื่อหาไฟล์ทั้งหมด
        data_path = Path(data_folder)
        for image_path in data_path.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in image_extensions:
                images.append(str(image_path))
        
        print(f"พบรูปภาพทั้งหมด {len(images)} รูป")
        return images
    
    def process_all_images(self):
        """ประมวลผลรูปภาพทั้งหมดและจัดเก็บผลลัพธ์"""
        images = self.get_all_images_from_data_folder()
        
        if not images:
            print("ไม่พบรูปภาพใน folder data")
            return
        
        # สถิติการจำแนก
        classification_stats = {category: 0 for category in self.id2label.values()}
        detailed_results = []
        
        print(f"เริ่มประมวลผลรูปภาพ {len(images)} รูป...")
        
        for i, image_path in enumerate(images, 1):
            print(f"ประมวลผลรูปที่ {i}/{len(images)}: {image_path}")
            
            # ทำนายหมวดหมู่
            predicted_category, results = self.predict_image(image_path)
            
            # อัปเดตสถิติ
            classification_stats[predicted_category] += 1
            
            # คัดลอกไฟล์ไปยัง folder ที่เหมาะสม
            filename = os.path.basename(image_path)
            destination_path = os.path.join(self.result_folder, predicted_category, filename)
            
            try:
                shutil.copy2(image_path, destination_path)
                print(f"  -> จำแนกเป็น: {predicted_category} (ความมั่นใจ: {results.get('confidence', 'N/A')}%)")
            except Exception as e:
                print(f"  -> เกิดข้อผิดพลาดในการคัดลอกไฟล์: {e}")
            
            # เก็บผลลัพธ์รายละเอียด
            detailed_results.append({
                "image_path": image_path,
                "predicted_category": predicted_category,
                "results": results
            })
        
        # บันทึกสถิติและผลลัพธ์
        self.save_results(classification_stats, detailed_results)
        
        # แสดงสรุปผลลัพธ์
        self.print_summary(classification_stats, len(images))
    
    def save_results(self, classification_stats, detailed_results):
        """บันทึกผลลัพธ์ลงไฟล์"""
        # บันทึกสถิติการจำแนก
        stats_file = os.path.join(self.result_folder, "classification_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(classification_stats, f, ensure_ascii=False, indent=2)
        
        # บันทึกผลลัพธ์รายละเอียด
        detailed_file = os.path.join(self.result_folder, "detailed_results.json")
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        print(f"\nบันทึกผลลัพธ์แล้ว:")
        print(f"  - สถิติการจำแนก: {stats_file}")
        print(f"  - ผลลัพธ์รายละเอียด: {detailed_file}")
    
    def print_summary(self, classification_stats, total_images):
        """แสดงสรุปผลลัพธ์"""
        print("\n" + "="*50)
        print("สรุปผลการจำแนกรูปภาพ")
        print("="*50)
        print(f"จำนวนรูปภาพทั้งหมด: {total_images}")
        print("-"*50)
        
        for category, count in classification_stats.items():
            percentage = (count / total_images) * 100 if total_images > 0 else 0
            print(f"{category:10}: {count:4} รูป ({percentage:5.1f}%)")
        
        print("="*50)
        print(f"รูปภาพถูกจัดเก็บใน folder: {self.result_folder}")

def main():
    """ฟังก์ชันหลัก"""
    print("เริ่มต้น NSFW Model...")
    
    # สร้าง instance ของ NSFWModel
    nsfw_model = NSFWModel()
    
    # ประมวลผลรูปภาพทั้งหมด
    nsfw_model.process_all_images()
    
    print("\nเสร็จสิ้นการประมวลผล!")

if __name__ == "__main__":
    main() 