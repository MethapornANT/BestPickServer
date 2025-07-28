import pandas as pd
import os
from datetime import datetime
import glob # สำหรับค้นหาไฟล์ด้วย wildcard

# --- การตั้งค่า ---
# กำหนดโฟลเดอร์ที่เก็บไฟล์ CSV (ควรเป็น directory เดียวกันกับโค้ดนี้)
CSV_FOLDER = './' 
OUTPUT_FOLDER = './comparison_results'
os.makedirs(OUTPUT_FOLDER, exist_ok=True) # สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์การเปรียบเทียบ

# Labels ที่ใช้ในการเปรียบเทียบ (ต้องตรงกับที่คุณใช้ในทั้งสองไฟล์)
LABELS_TO_COMPARE = ['Normal', 'Hentai', 'Porn', 'Sexy', 'Anime']

# --- ฟังก์ชันหลักสำหรับการเปรียบเทียบ ---
def compare_nsfw_predictions(custom_csv_path, hf_csv_path):
    """
    ฟังก์ชันสำหรับอ่านไฟล์ CSV สองไฟล์ (จากโมเดล Custom และ Hugging Face)
    และเปรียบเทียบผลการทำนายแบบละเอียด
    """
    print(f"กำลังอ่านไฟล์ Custom Model: {os.path.basename(custom_csv_path)} งับ")
    print(f"กำลังอ่านไฟล์ Hugging Face Model: {os.path.basename(hf_csv_path)} งับ")

    try:
        df_custom = pd.read_csv(custom_csv_path)
        df_hf = pd.read_csv(hf_csv_path)
    except FileNotFoundError:
        print("❌ Error: ไม่พบไฟล์ CSV ที่ระบุ โปรดตรวจสอบ Path ให้ถูกต้องนะงับ")
        return None
    except Exception as e:
        print(f"💔 เกิดข้อผิดพลาดในการอ่านไฟล์ CSV: {e} งับ")
        return None

    # เตรียม DataFrame สำหรับเก็บผลลัพธ์การเปรียบเทียบ
    comparison_results = []

    # รวม DataFrame โดยใช้ Image_Name เป็นคีย์
    # ใช้ merge แทน concat เพราะต้องการรวมข้อมูลในแถวเดียวกัน
    # กำหนด suffixes เพื่อแยกคอลัมน์จากแต่ละโมเดล
    merged_df = pd.merge(
        df_custom, 
        df_hf, 
        on='Image_Name', 
        how='inner', # ใช้ inner เพื่อเอาเฉพาะรูปที่มีอยู่ในทั้งสองไฟล์
        suffixes=('_Custom', '_HF')
    )

    if merged_df.empty:
        print("⚠️ ไม่มีรูปภาพที่ตรงกันในทั้งสองไฟล์ CSV เลยงับ! ตรวจสอบชื่อรูปภาพหน่อยนะ")
        return None

    for index, row in merged_df.iterrows():
        img_name = row['Image_Name']
        row_data = {'Image_Name': img_name}

        # เปรียบเทียบผลลัพธ์ NSFW โดยรวม
        is_nsfw_custom = row.get('is_NSFW_Custom', 'N/A')
        is_nsfw_hf = row.get('is_NSFW_HF', 'N/A')
        
        row_data['is_NSFW_Custom'] = is_nsfw_custom
        row_data['is_NSFW_HF'] = is_nsfw_hf
        row_data['NSFW_Match'] = '✅ Match' if is_nsfw_custom == is_nsfw_hf else '❌ Mismatch'

        print(f"\n--- เปรียบเทียบผลลัพธ์สำหรับ: {img_name} ---")
        print(f"  - NSFW (Custom): {is_nsfw_custom}")
        print(f"  - NSFW (HF): {is_nsfw_hf}")
        print(f"  -> สรุป NSFW: {row_data['NSFW_Match']}")

        # เปรียบเทียบเปอร์เซ็นต์ของแต่ละ Label
        for label in LABELS_TO_COMPARE:
            custom_score_col = f'{label}_%_Custom'
            hf_score_col = f'{label}_%_HF'

            custom_score = row.get(custom_score_col, 'N/A')
            hf_score = row.get(hf_score_col, 'N/A')
            
            row_data[f'{label}_%_Custom'] = custom_score
            row_data[f'{label}_%_HF'] = hf_score

            # คำนวณความต่าง (ถ้าเป็นตัวเลข)
            score_diff = 'N/A'
            try:
                if isinstance(custom_score, (int, float)) and isinstance(hf_score, (int, float)):
                    score_diff = round(abs(custom_score - hf_score), 2)
            except:
                pass # กรณี N/A หรือ Type Error

            row_data[f'{label}_Diff'] = score_diff

            print(f"  - {label}: Custom={custom_score:.2f}% | HF={hf_score:.2f}% | Diff={score_diff}")
            
        comparison_results.append(row_data)

    df_comparison = pd.DataFrame(comparison_results)
    return df_comparison

# --- ส่วนของการเลือกไฟล์และรัน ---
def select_csv_file(prefix):
    """ให้ผู้ใช้เลือกไฟล์ CSV ที่ต้องการจากรายการ"""
    files = glob.glob(os.path.join(CSV_FOLDER, f'{prefix}*.csv'))
    
    if not files:
        print(f"❌ ไม่พบไฟล์ CSV ที่ขึ้นต้นด้วย '{prefix}' ในโฟลเดอร์ '{CSV_FOLDER}' เลยงับ")
        return None
    
    print(f"\n--- พบไฟล์ CSV ที่ขึ้นต้นด้วย '{prefix}' ดังนี้งับ ---")
    for i, f in enumerate(files):
        print(f"{i+1}. {os.path.basename(f)}")
    
    while True:
        try:
            choice = input(f"กรุณาเลือกไฟล์ '{prefix}' ที่ต้องการเปรียบเทียบ (ป้อนตัวเลข): ")
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
            else:
                print("⚠️ ตัวเลขไม่อยู่ในรายการที่เลือกได้งับ ลองใหม่นะ")
        except ValueError:
            print("⚠️ กรุณาป้อนเป็นตัวเลขนะงับ")

if __name__ == "__main__":
    print("\n✨ เริ่มต้นการเปรียบเทียบผลลัพธ์ NSFW ของ 2 โมเดล ✨")

    # ให้ผู้ใช้เลือกไฟล์ Custom Model CSV
    custom_model_csv = select_csv_file('custom_nsfw_results')
    if custom_model_csv is None:
        exit()

    # ให้ผู้ใช้เลือกไฟล์ Hugging Face Model CSV
    hf_model_csv = select_csv_file('hf_nsfw_results')
    if hf_model_csv is None:
        exit()

    # ทำการเปรียบเทียบ
    final_comparison_df = compare_nsfw_predictions(custom_model_csv, hf_model_csv)

    if final_comparison_df is not None and not final_comparison_df.empty:
        # บันทึกผลลัพธ์การเปรียบเทียบลง CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filepath = os.path.join(OUTPUT_FOLDER, f"nsfw_comparison_summary_{timestamp}.csv")
        final_comparison_df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
        print(f"\n🎉 บันทึกผลการเปรียบเทียบแบบละเอียดลงในไฟล์: {output_filepath} แล้วงับ!")
    else:
        print("\n💔 ไม่สามารถสร้างไฟล์ผลลัพธ์การเปรียบเทียบได้ (อาจไม่มีข้อมูลหรือเกิดข้อผิดพลาด) งับ")

    print("\n--- การเปรียบเทียบเสร็จสิ้นแล้วงับ! ---")