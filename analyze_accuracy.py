import os
import json
from pathlib import Path

def analyze_model_accuracy():
    """วิเคราะห์ความแม่นยำของโมเดล NSFW"""
    
    result_folder = "result"
    categories = ["anime", "hentai", "normal", "porn", "sexy"]
    
    # สถิติการจำแนก
    total_images = 0
    correct_predictions = 0
    category_stats = {}
    
    for category in categories:
        category_folder = os.path.join(result_folder, category)
        if not os.path.exists(category_folder):
            continue
            
        category_total = 0
        category_correct = 0
        
        # นับไฟล์ในแต่ละ folder
        for filename in os.listdir(category_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                category_total += 1
                total_images += 1
                
                # ตรวจสอบว่าชื่อไฟล์ตรงกับชื่อ folder หรือไม่
                file_prefix = filename.split('_')[0].lower()
                if file_prefix == category:
                    category_correct += 1
                    correct_predictions += 1
        
        category_stats[category] = {
            'total': category_total,
            'correct': category_correct,
            'accuracy': (category_correct / category_total * 100) if category_total > 0 else 0
        }
    
    # คำนวณความแม่นยำรวม
    overall_accuracy = (correct_predictions / total_images * 100) if total_images > 0 else 0
    
    # แสดงผลลัพธ์
    print("="*60)
    print("ผลการวิเคราะห์ความแม่นยำของโมเดล NSFW")
    print("="*60)
    print(f"จำนวนรูปภาพทั้งหมด: {total_images}")
    print(f"จำนวนที่จำแนกถูกต้อง: {correct_predictions}")
    print(f"ความแม่นยำรวม: {overall_accuracy:.2f}%")
    print("-"*60)
    
    for category, stats in category_stats.items():
        print(f"{category:10}: {stats['correct']:4}/{stats['total']:4} รูป ({stats['accuracy']:5.1f}%)")
    
    print("-"*60)
    print(f"ความแม่นยำรวม: {overall_accuracy:.2f}%")
    print("="*60)
    
    # บันทึกผลลัพธ์
    results = {
        "total_images": total_images,
        "correct_predictions": correct_predictions,
        "overall_accuracy": overall_accuracy,
        "category_stats": category_stats
    }
    
    with open(os.path.join(result_folder, "accuracy_analysis.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nบันทึกผลการวิเคราะห์แล้ว: {os.path.join(result_folder, 'accuracy_analysis.json')}")
    
    return results

def show_detailed_misclassifications():
    """แสดงรายละเอียดการจำแนกผิด"""
    
    result_folder = "result"
    categories = ["anime", "hentai", "normal", "porn", "sexy"]
    
    print("\n" + "="*60)
    print("รายละเอียดการจำแนกผิด")
    print("="*60)
    
    for category in categories:
        category_folder = os.path.join(result_folder, category)
        if not os.path.exists(category_folder):
            continue
            
        misclassified = []
        
        for filename in os.listdir(category_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                file_prefix = filename.split('_')[0].lower()
                if file_prefix != category:
                    misclassified.append(filename)
        
        if misclassified:
            print(f"\n{category.upper()} folder (จำแนกผิด {len(misclassified)} รูป):")
            print("-" * 40)
            
            # จัดกลุ่มตามประเภทที่ควรจะเป็น
            misclassification_groups = {}
            for filename in misclassified:
                actual_category = filename.split('_')[0].lower()
                if actual_category not in misclassification_groups:
                    misclassification_groups[actual_category] = []
                misclassification_groups[actual_category].append(filename)
            
            for actual_cat, files in misclassification_groups.items():
                print(f"  ควรเป็น {actual_cat}: {len(files)} รูป")
                if len(files) <= 5:  # แสดงตัวอย่างไม่เกิน 5 รูป
                    for file in files[:5]:
                        print(f"    - {file}")
                else:
                    for file in files[:3]:
                        print(f"    - {file}")
                    print(f"    ... และอีก {len(files)-3} รูป")

if __name__ == "__main__":
    # วิเคราะห์ความแม่นยำ
    results = analyze_model_accuracy()
    
    # แสดงรายละเอียดการจำแนกผิด
    show_detailed_misclassifications() 