# -*- coding: utf-8 -*-
"""
ดาวน์โหลดภาพจาก Google ตาม ProductName แล้วอัปเดตลง posts.photo_url (JSON)
- ไม่ใช้ Bing API
- ใช้ icrawler.builtin.GoogleImageCrawler
- คัดกรองรูปไม่กาก (เช็คขนาดขั้นต่ำและประเภทไฟล์)
- แปลงเป็น .jpg คุณภาพโอเค
- บันทึกไฟล์ไปที่ ./uploads/{post_id}_{1..3}.jpg
- อัปเดต DB เป็น JSON_ARRAY('/uploads/..', '/uploads/..', '/uploads/..')
- แสดง progress ชัดเจน
"""

import os
import re
import time
import random
import shutil
import json
from pathlib import Path
from io import BytesIO

from PIL import Image
from icrawler.builtin import GoogleImageCrawler
from tqdm import tqdm
import mysql.connector

# ----------------------------- CONFIG -----------------------------
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "1234",
    "database": "bestpick",
    "charset": "utf8mb4",
    "use_unicode": True,
}
UPLOAD_DIR = Path("uploads")  # ไฟล์จริง
WEB_PREFIX = "/uploads"       # path ที่เก็บลง DB
MIN_W, MIN_H = 640, 640       # กันรูปกาก
QUALITY = 88                  # JPEG quality
SLEEP_BETWEEN = (2.0, 4.5)    # หน่วงกันโดน block

# เลือกกลุ่มโพสต์ใหม่ที่เคยให้กูสร้างไว้ (แก้ช่วง id ได้ตามจริงของมึง)
POST_ID_START = 1380131
POST_ID_END   = 1380210

# -----------------------------------------------------------------

def ensure_upload_dir():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def connect_db():
    conn = mysql.connector.connect(**DB_CONFIG)
    conn.autocommit = False
    return conn

def get_target_posts(conn):
    """
    เลือกเฉพาะโพสต์ช่วง id ที่ระบุ และยังไม่มีรูป (photo_url ว่าง)
    """
    sql = """
        SELECT id, ProductName
        FROM posts
        WHERE id BETWEEN %s AND %s
          AND (photo_url IS NULL OR JSON_LENGTH(photo_url) = 0)
    """
    cur = conn.cursor()
    cur.execute(sql, (POST_ID_START, POST_ID_END))
    rows = cur.fetchall()
    cur.close()
    return [{"id": r[0], "product": r[1] or ""} for r in rows]

def sanitize_filename(s):
    s = re.sub(r"[^\w\-_.]", "_", s, flags=re.UNICODE)
    return s[:64]

def convert_to_jpeg(src_path: Path, dst_path: Path):
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            im.save(dst_path, "JPEG", quality=QUALITY, optimize=True, progressive=True)
        return True, ""
    except Exception as e:
        return False, str(e)

def is_large_enough(p: Path):
    try:
        with Image.open(p) as im:
            w, h = im.size
            return (w >= MIN_W and h >= MIN_H)
    except Exception:
        return False

def crawl_google_images(keywords, tmp_dir: Path, max_num=12):
    """
    ลองโหลดภาพจาก Google ด้วย icrawler
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    crawler = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=2,
        downloader_threads=3,
        storage={"root_dir": str(tmp_dir)},
    )
    # พอให้โอกาส Google หายใจ
    time.sleep(random.uniform(*SLEEP_BETWEEN))

    try:
        crawler.crawl(
            keyword=keywords,
            max_num=max_num,
            filters={"size": "large", "type": "photo"},
            file_idx_offset=0
        )
    except Exception:
        # บางทีเจอ rate limit/robot, ก็ให้ผ่านไป เดี๋ยว fallback ช่วย
        pass

def pick_top3_and_convert(tmp_dir: Path, post_id: int):
    """
    คัดไฟล์ภาพจาก tmp_dir เลือก 3 ภาพที่รอดเงื่อนไข แล้วแปลง/ย้ายไป uploads
    """
    files = sorted([p for p in tmp_dir.glob("**/*") if p.is_file()])
    chosen = []

    for p in files:
        # skip ไฟล์ขยะ
        if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue
        if not is_large_enough(p):
            continue

        # แปลงเป็น .jpg
        dst_name = f"{post_id}_{len(chosen)+1}.jpg"
        dst_path = UPLOAD_DIR / dst_name
        ok, err = convert_to_jpeg(p, dst_path)
        if not ok:
            continue

        chosen.append(dst_path)
        if len(chosen) == 3:
            break

    return chosen

def google_fetch_three(post_id: int, product_name: str):
    """
    พยายามหลายคีย์เวิร์ดจนกว่าจะได้ครบ 3
    """
    safe_kw = product_name.strip()
    if not safe_kw:
        safe_kw = f"product {post_id}"

    # คีย์เวิร์ด fallback ไล่ยิงให้ครบ 3
    candidates = [
        safe_kw,
        f"{safe_kw} product",
        f"{safe_kw} review",
        f"{safe_kw} official",
        f"{safe_kw} photo",
        f"{safe_kw} -wallpaper -logo -icon",
    ]

    tmp_root = Path(".cache_google") / str(post_id)
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    gathered = []
    tried = 0

    for kw in candidates:
        tried += 1
        sub_tmp = tmp_root / sanitize_filename(kw) / "raw"
        crawl_google_images(kw, sub_tmp, max_num=12)
        picked = pick_top3_and_convert(sub_tmp, post_id)

        # รวมผล (อย่าเกิน 3)
        for p in picked:
            if p not in gathered:
                gathered.append(p)
            if len(gathered) == 3:
                break

        if len(gathered) == 3:
            break

    # เก็บกวาด tmp
    shutil.rmtree(tmp_root, ignore_errors=True)
    return gathered, tried

def update_photo_urls(conn, post_id: int, final_paths):
    """
    final_paths: list[Path] -> เขียน JSON_ARRAY('/uploads/a.jpg', '/uploads/b.jpg', '/uploads/c.jpg')
    """
    web_paths = [str(Path(WEB_PREFIX) / p.name) for p in final_paths]
    # กรณีเผื่อได้น้อยกว่า 3 แต่ยังอยากอัปเดตเท่าที่มี (ปกติเราบังคับให้ครบ 3 อยู่)
    placeholders = ", ".join(["%s"] * len(web_paths))
    sql = f"UPDATE posts SET photo_url = JSON_ARRAY({placeholders}) WHERE id = %s"
    params = web_paths + [post_id]
    cur = conn.cursor()
    cur.execute(sql, params)
    cur.close()

def main():
    ensure_upload_dir()
    conn = connect_db()
    posts = get_target_posts(conn)

    if not posts:
        print("ไม่มีโพสต์เป้าหมายให้เติมรูป (photo_url ว่าง) ในช่วง id ที่กำหนด โชคดีของกู…")
        return

    print(f"จะอัปเดตรูปให้ {len(posts)} โพสต์ (id {POST_ID_START}..{POST_ID_END})")
    ok_count = 0
    fail = []

    for item in tqdm(posts, desc="กำลังลุย Google หารูป", unit="post"):
        pid = item["id"]
        pname = item["product"] or ""
        try:
            imgs, tried = google_fetch_three(pid, pname)
            if len(imgs) < 3:
                fail.append((pid, pname, f"ได้ {len(imgs)}/3 หลังลอง {tried} คีย์เวิร์ด"))
                continue
            update_photo_urls(conn, pid, imgs)
            conn.commit()
            ok_count += 1
            print(f"\n[OK] id={pid} '{pname}' -> {[str(Path(WEB_PREFIX)/p.name) for p in imgs]}")
            time.sleep(random.uniform(*SLEEP_BETWEEN))
        except Exception as e:
            conn.rollback()
            fail.append((pid, pname, str(e)))

    print("\nสรุปงาน:")
    print(f"- อัปเดตสำเร็จ: {ok_count}/{len(posts)} โพสต์")
    if fail:
        print(f"- ตกค้าง: {len(fail)} โพสต์ (ตามรายการด้านล่าง)")
        for pid, pname, reason in fail:
            print(f"  * id={pid} '{pname}' --> {reason}")
    else:
        print("- ไม่มีโพสต์ตกค้าง กูก็ยังพอมีความสุขกับจักรวาลนี้อยู่บ้าง")

if __name__ == "__main__":
    main()
