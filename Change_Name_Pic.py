# rename_safe.py
import os
import re
import uuid
import shutil
import argparse

folder_path = "./data4/normal"  # ปรับตามต้องการ

parser = argparse.ArgumentParser(description="Replace _1_.._16_ -> _13_.._28_ in filenames (safe two-phase rename).")
parser.add_argument("--apply", action="store_true", help="Apply changes (otherwise dry-run).")
parser.add_argument("--backup", action="store_true", help="Make a backup copy before applying.")
args = parser.parse_args()

DRY_RUN = not args.apply

pattern = re.compile(r'_(\d{1,2})_')

# 1) ถ้าขอ backup ให้ทำก่อน (จะสร้างโฟลเดอร์ backup_<timestamp>)
if args.backup and not DRY_RUN:
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = folder_path.rstrip("/\\") + f"_backup_{ts}"
    print(f"[INFO] สร้าง backup: {folder_path} -> {backup_path}")
    shutil.copytree(folder_path, backup_path)
    print("[INFO] Backup สำเร็จ")

# อ่านไฟล์
try:
    file_list = os.listdir(folder_path)
except FileNotFoundError:
    print(f"[ERROR] โฟลเดอร์ไม่พบ: {folder_path}")
    raise SystemExit(1)

tmp_map = {}  # tmp_path -> final_filename
planned = []

# ขั้นที่ 1: สร้างชื่อชั่วคราวสำหรับไฟล์ที่จะเปลี่ยน
for filename in file_list:
    m = pattern.search(filename)
    if not m:
        continue
    num = int(m.group(1))
    if 1 <= num <= 16:
        new_num = num + 12
        new_filename = filename[:m.start()] + f"_{new_num}_" + filename[m.end():]
        if new_filename == filename:
            continue
        old_path = os.path.join(folder_path, filename)
        tmp_name = filename + ".renametmp." + uuid.uuid4().hex
        tmp_path = os.path.join(folder_path, tmp_name)
        planned.append((filename, tmp_name, new_filename))
        if DRY_RUN:
            print(f"[DRY] จะเปลี่ยนชั่วคราว: {filename} -> {tmp_name} -> เป้าหมาย: {new_filename}")
        else:
            os.rename(old_path, tmp_path)
        tmp_map[tmp_path] = new_filename

if not planned:
    print("[INFO] ไม่มีไฟล์ที่ต้องเปลี่ยน (pattern _1_.._16_ ไม่เจอ)")
else:
    print(f"[INFO] เจอ {len(planned)} ไฟล์ ที่จะเปลี่ยน (DRY_RUN={DRY_RUN})")

# ขั้นที่ 2: เปลี่ยนชื่อจากชั่วคราวเป็นชื่อสุดท้าย (ปลอดภัยต่อการทับไฟล์)
done = []
conflicts = 0
for tmp_path, final_name in tmp_map.items():
    final_path = os.path.join(folder_path, final_name)
    if DRY_RUN:
        print(f"[DRY] จะเปลี่ยนสุดท้าย: {os.path.basename(tmp_path)} -> {final_name}")
    else:
        if os.path.exists(final_path):
            # ป้องกันทับชื่อเดิม (should be rare because we used tmp phase)
            base, ext = os.path.splitext(final_name)
            safe_name = f"{base}.conflict.{uuid.uuid4().hex}{ext}"
            final_path = os.path.join(folder_path, safe_name)
            conflicts += 1
            final_name = safe_name
        os.rename(tmp_path, final_path)
        done.append(final_name)
        print(f"[OK] เปลี่ยน: {os.path.basename(tmp_path)} -> {final_name}")

if not DRY_RUN:
    print(f"[DONE] รวม {len(done)} ไฟล์เปลี่ยนชื่อ เสร็จแล้ว — conflicts: {conflicts}")
else:
    print("[DRY] เสร็จการตรวจ (ยังไม่ได้เปลี่ยนจริง)")
