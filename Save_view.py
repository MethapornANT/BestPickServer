import pandas as pd
from sqlalchemy import create_engine
import sys

# ===== Global config =====
# แก้ไข user, password, host, และ database name ตามของคุณ
DB_URI = 'mysql+mysqlconnector://root:1234@localhost/bestpick'

# ===== ชื่อ Views และชื่อไฟล์ที่จะ Export =====
views_to_export = {
    'contentbasedview': 'content_based_data.csv',
    'collaborativeview': 'collaborative_data.csv'
}

def export_views_to_csv():
    """
    ฟังก์ชันสำหรับเชื่อมต่อฐานข้อมูล, ดึงข้อมูลจาก Views,
    และบันทึกเป็นไฟล์ CSV
    """
    try:
        # สร้าง engine สำหรับเชื่อมต่อฐานข้อมูล
        engine = create_engine(DB_URI)
        print("🚀 เชื่อมต่อฐานข้อมูลสำเร็จ!")

        # วนลูปเพื่อดึงข้อมูลและบันทึกไฟล์
        for view_name, file_name in views_to_export.items():
            print(f"กำลังดึงข้อมูลจาก '{view_name}'...")
            
            # เขียน SQL query
            query = f"SELECT * FROM {view_name}"
            
            # ดึงข้อมูลมาเก็บใน pandas DataFrame
            df = pd.read_sql(query, engine)
            
            # บันทึก DataFrame เป็นไฟล์ CSV
            # ใช้ encoding='utf-8-sig' เพื่อให้เปิดใน Excel แล้วอ่านภาษาไทยได้ถูกต้อง
            df.to_csv(file_name, index=False, encoding='utf-8-sig')
            
            print(f"✅ บันทึกข้อมูลจาก '{view_name}' ไปยัง '{file_name}' เรียบร้อยแล้ว\n")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # ปิดการเชื่อมต่อ
        if 'engine' in locals():
            engine.dispose()
            print("🔗 ปิดการเชื่อมต่อฐานข้อมูลแล้ว")

# ===== สั่งรันฟังก์ชัน =====
if __name__ == '__main__':
    export_views_to_csv()