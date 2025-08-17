import asyncio
import os
import shutil
import hashlib
from aiohttp import ClientSession, ClientTimeout
from urllib.parse import urlparse, urlencode
from playwright.async_api import async_playwright
import time

# ================== CONFIG =====================
DATA_FOLDER = "data"                # โฟลเดอร์หลักเก็บรูป
TIMEOUT_DURATION = 7                    # timeout ต่อ request (วินาที)
CLEAR_FOLDER_BEFORE_RUN = True          # True = ลบไฟล์ภาพเก่าทั้งหมดก่อนรัน, False = โหลดต่อจากเดิม
HEADLESS_BROWSER = True                # False = เปิด browser ให้เห็น, True = ซ่อน browser
CATEGORY_SEARCHES = {
    "anime": ["anime", "anime girl", "anime boy", "anime love", "anime kiss", "cartoon charater", "anime AI", "anime cute", "anime beautiful"],
    "hentai": ["hentai", "hentai girl", "hentai boy", "hentai nude", "cartoon porn", "hentai fuck", "hentai cute", "hentai beautiful"],
    "normal": ["boy", "girl", "women", "man", "cat" , "nature", "athlete", "footballer", "dog", "animal"],
    "porn": ["porn xxx", "nude porn", "pussy xxx", "cock xxx", "gay porn", "sex porn", "hardcore porn", "adult porn", "lesbian porn"],
    "sexy": ["bikini girl", "sexy girl", "bikini women", "sexy", "sexy man", "cartoon sexy", "underwear sexy", "hot girl", "lingeries", "no bra sexy"]
}
MAX_IMAGES_PER_CATEGORY = 3000          # จำนวนรูปต่อหมวดหมู่
MAX_RELATED_DEPTH = 3                   # เพิ่มความลึกของ related search
BATCH_SIZE = 30                         # โหลดรูปพร้อมกันกี่รูป (เพิ่มความเร็ว)
# ===============================================

def calculate_image_hash(image_data: bytes) -> str:
    return hashlib.md5(image_data).hexdigest()

def clear_all_category_images():
    for category in CATEGORY_SEARCHES:
        folder_path = os.path.join(DATA_FOLDER, category)
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"ลบไฟล์ภาพทั้งหมดใน: {folder_path}")

def ensure_folder_exists(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"สร้างโฟลเดอร์: {folder_path}")

def count_actual_files_in_folder(folder_path):
    """นับไฟล์จริงในโฟลเดอร์"""
    if not os.path.exists(folder_path):
        return 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    count = 0
    
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in image_extensions and os.path.getsize(file_path) > 1000:
                count += 1
    
    return count

def get_image_summary():
    """สรุปจำนวนภาพในแต่ละโฟลเดอร์"""
    picdata_path = "PicData"
    if not os.path.exists(picdata_path):
        return {}
    
    summary = {}
    categories = ["anime", "hentai", "normal", "porn", "sexy"]
    
    for category in categories:
        category_path = os.path.join(picdata_path, category)
        count = count_actual_files_in_folder(category_path)
        summary[category] = count
    
    return summary

def clean_corrupted_images():
    """ลบไฟล์ภาพเสีย"""
    picdata_path = "PicData"
    if not os.path.exists(picdata_path):
        return 0
    
    cleaned_count = 0
    categories = ["anime", "hentai", "normal", "porn", "sexy"]
    
    for category in categories:
        category_path = os.path.join(picdata_path, category)
        if not os.path.exists(category_path):
            continue
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in image_extensions:
                    try:
                        file_size = os.path.getsize(file_path)
                        if file_size < 1000:  # ไฟล์เสีย (น้อยกว่า 1KB)
                            os.remove(file_path)
                            cleaned_count += 1
                    except:
                        pass
    
    return cleaned_count

async def disable_safesearch(page):
    try:
        safesearch_btn = await page.query_selector('div[aria-pressed][role="button"]:has-text("ปิดอยู่")')
        if safesearch_btn:
            print("พบ SafeSearch: ปิดอยู่ → กำลังปิด...")
            await safesearch_btn.click()
            await asyncio.sleep(1)
            await page.reload()
            print("ปิด SafeSearch สำเร็จ รีเฟรชหน้าแล้ว")
            return True
    except Exception:
        pass
    return False

def print_progress_bar(current, total, query, bar_length=30):
    percent = current / total if total else 0
    bar = '#' * int(bar_length * percent) + '-' * (bar_length - int(bar_length * percent))
    print(f"\r[{bar}] {percent*100:.1f}% ({current}/{total}) | {query}", end='', flush=True)

async def get_related_searches(page):
    related = set()
    try:
        elements = await page.query_selector_all('a[aria-label^="ค้นหาเพิ่มเติมเกี่ยวกับ"]')
        for el in elements:
            text = await el.inner_text()
            if text:
                related.add(text.strip())
        if not related:
            related_box = await page.query_selector('div:has(h2:has-text("การค้นหาที่เกี่ยวข้อง"))')
            if related_box:
                links = await related_box.query_selector_all('a')
                for link in links:
                    text = await link.inner_text()
                    if text:
                        related.add(text.strip())
    except Exception:
        pass
    return list(related)

async def click_last_related_search(page):
    try:
        related_box = await page.query_selector('div:has(h2:has-text("การค้นหาที่เกี่ยวข้อง"))')
        if related_box:
            links = await related_box.query_selector_all('a')
            if links:
                await links[-1].click()
                await asyncio.sleep(0.5)
                return True
    except Exception:
        pass
    return False

async def click_load_more(page):
    # ฟังก์ชันนี้จะเลื่อนลงไปยังภาพสุดท้าย แล้วหาและคลิกปุ่ม 'ดูเพิ่มเติม' (ถ้ามี)
    try:
        # เลื่อนลงไปยังภาพสุดท้าย (ฝั่งขวา)
        await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
        await asyncio.sleep(0.3)
        # หาโดยข้อความ
        btn = await page.query_selector('div:has-text("ดูเพิ่มเติม")')
        if not btn:
            # หาโดย class (เช่น CCYCud)
            btn = await page.query_selector('div.CCYCud')
        if btn:
            await btn.click()
            await asyncio.sleep(0.5)
            return True
    except Exception:
        pass
    return False

async def click_load_more_in_right_panel(page):
    try:
        # scroll panel ด้านขวาลงสุด (selector ใหม่)
        panel = await page.query_selector('div.BIB1wf.ElehLd.fHE6De.Emjfj d')
        used_selector = 'div.BIB1wf.ElehLd.fHE6De.Emjfj d'
        if not panel:
            # fallback selector เดิม
            panel = await page.query_selector('div.hh1Ztf.ip4nvd.k4o2Hc')
            used_selector = 'div.hh1Ztf.ip4nvd.k4o2Hc'
        if panel:
            print(f'  [LOG] scroll panel ขวาด้วย selector: {used_selector}')
            await page.evaluate('(el) => { el.scrollTop = el.scrollHeight }', panel)
            await asyncio.sleep(0.7)
            # หาและคลิกปุ่ม 'ดูเพิ่มเติม' ใน panel ด้านขวา
            btn = await page.query_selector('div.CCYCud, div.CCYCud.A7KIJf')
            if btn:
                print('  [LOG] ➡️ เจอปุ่ม "ดูเพิ่มเติม" ใน panel ด้านขวา กำลังกด...')
                await btn.click()
                await asyncio.sleep(1.2)
                return True
            else:
                print('  [LOG] ❗ ไม่เจอปุ่ม "ดูเพิ่มเติม" ใน panel ด้านขวา')
        else:
            print('  [LOG] ❗ ไม่เจอ panel ด้านขวา (ทั้ง selector ใหม่และเก่า)')
    except Exception as e:
        print(f'  [LOG] ❗ Error scroll/click load more in right panel: {e}')
    return False

async def download_image(session, img_url, file_path, downloaded_hashes):
    try:
        async with session.get(img_url, timeout=5) as response:
            if response.status == 200:
                image_data = await response.read()
                image_hash = calculate_image_hash(image_data)
                if image_hash in downloaded_hashes:
                    return False
                file_path = os.path.splitext(file_path)[0] + ".jpg"
                with open(file_path, "wb") as f:
                    f.write(image_data)
                # ตรวจสอบว่าไฟล์บันทึกสำเร็จและมีขนาดพอ
                if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:
                    downloaded_hashes.add(image_hash)
                    return True
                else:
                    # ลบไฟล์เสีย
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    return False
    except Exception:
        pass
    return False

async def scroll_to_bottom(page):
    previous_height = await page.evaluate("document.body.scrollHeight")
    for _ in range(1):  # scroll 1 รอบเพื่อความเร็ว
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(0.2)
        new_height = await page.evaluate("document.body.scrollHeight")
        if new_height == previous_height:
            break
        previous_height = new_height

async def scroll_right_panel(page):
    # หา div panel ด้านขวา (เช่น class 'hh1Ztf ip4nvd k4o2Hc')
    panel = await page.query_selector('div.hh1Ztf.ip4nvd.k4o2Hc')
    if panel:
        await page.evaluate('(el) => { el.scrollTop = el.scrollHeight }', panel)
        await asyncio.sleep(0.5)

# --- ลบฟังก์ชัน click_load_more_in_panel ---

async def scrape_images_for_category(category, search_terms, max_images, max_related_depth, session, page):
    category_folder = os.path.join(DATA_FOLDER, category)
    ensure_folder_exists(category_folder)
    downloaded_hashes = set()
    
    # คำนวณ quota ให้กระจายเท่าๆ กัน
    quota_per_query = max_images // len(search_terms)
    quotas = [quota_per_query] * len(search_terms)
    # กระจายเศษให้คำค้นแรก
    remaining = max_images - (quota_per_query * len(search_terms))
    if remaining > 0:
        quotas[0] += remaining
    
    total_downloaded = 0
    category_start_time = time.time()
    
    for query_idx, (query, quota) in enumerate(zip(search_terms, quotas)):
        query_start_time = time.time()
        used_queries = set()
        queue = [(query, 0)]  # (query, depth)
        downloaded_count = 0
        error_reason = None
        max_attempts = 3  # พยายาม 3 ครั้งต่อคำค้น
        
        for attempt in range(max_attempts):
            if downloaded_count >= quota:
                break
                
            while queue and downloaded_count < quota:
                current_query, depth = queue.pop(0)
                if current_query in used_queries:
                    continue
                used_queries.add(current_query)
                
                # แสดง progress bar ของแต่ละ query
                print(f"\n[หมวด {category}] คำค้น: {query} | คำค้นปัจจุบัน: {current_query} (depth {depth}) | quota {quota} | พยายามครั้งที่ {attempt+1}")
                query_params = urlencode({"q": current_query, "tbm": "isch"})
                search_url = f"https://www.google.com/search?{query_params}"
                await page.goto(search_url)
                await asyncio.sleep(0.2)
                await disable_safesearch(page)
                
                # --- scroll grid หลักซ้ำ ๆ เพื่อโหลดภาพใหม่ ---
                seen_img_count = 0
                max_scroll_rounds = 40  # เพิ่มจำนวนรอบ scroll
                for scroll_round in range(max_scroll_rounds):
                    await scroll_to_bottom(page)
                    await asyncio.sleep(0.5)
                    try:
                        await page.wait_for_selector('div[data-id="mosaic"]', timeout=3000)
                    except Exception:
                        error_reason = "ไม่พบ grid หลักของภาพ"
                        break
                    image_elements = await page.query_selector_all('div[data-attrid="images universal"]')
                    if len(image_elements) == seen_img_count:
                        break
                    
                    # ไล่คลิกภาพใหม่ที่ยังไม่ได้โหลด
                    img_idx = seen_img_count
                    while img_idx < len(image_elements) and downloaded_count < quota:
                        tasks = []
                        batch_size = min(BATCH_SIZE, quota - downloaded_count, len(image_elements) - img_idx)
                        file_name_indices = []
                        for b in range(batch_size):
                            image_element = image_elements[img_idx]
                            img_idx += 1
                            try:
                                await image_element.click()
                                await page.wait_for_selector("img.sFlh5c.FyHeAf.iPVvYb[jsaction]", timeout=5000)
                                img_tag = await page.query_selector("img.sFlh5c.FyHeAf.iPVvYb[jsaction]")
                                if not img_tag:
                                    continue
                                img_url = await img_tag.get_attribute("src")
                                if not img_url or img_url.startswith("data:"):
                                    continue
                                file_name = f"{category}_{query_idx+1}_{downloaded_count + len(tasks) + 1:04d}.jpg"
                                file_path = os.path.join(category_folder, file_name)
                                tasks.append(download_image(session, img_url, file_path, downloaded_hashes))
                                file_name_indices.append(file_name)
                            except Exception:
                                continue
                        if tasks:
                            results = await asyncio.gather(*tasks)
                            for i, success in enumerate(results):
                                if success:
                                    downloaded_count += 1
                                    print_progress_bar(downloaded_count, quota, query)
                    seen_img_count = len(image_elements)
                    if downloaded_count >= quota:
                        break
                
                print()  # ขึ้นบรรทัดใหม่หลังจบ progress bar
                
                # ถ้า quota ยังไม่ครบหลัง scroll grid ครบ → ไป related search (หรือดูเพิ่มเติมใน panel ขวา)
                if downloaded_count < quota:
                    related = []
                    if depth < max_related_depth:
                        related = await get_related_searches(page)
                        for rel in related:
                            if rel not in used_queries and all(rel != q for q, _ in queue):
                                queue.append((rel, depth + 1))
                    if not related:
                        # คลิกภาพสุดท้ายใน grid เพื่อเปิด panel ขวา
                        if image_elements:
                            try:
                                await image_elements[-1].click()
                                await asyncio.sleep(0.7)
                            except Exception:
                                pass
                        # scroll panel ขวา + กดดูเพิ่มเติม
                        load_more_success = await click_load_more_in_right_panel(page)
                        if load_more_success:
                            continue  # กลับไป scroll grid หลักใหม่
                        else:
                            error_reason = "ไม่พบ related search และไม่มีปุ่มดูเพิ่มเติมใน panel ขวา"
                            break
            
            # ถ้ายังไม่ครบ quota และยังมีคำค้นหาอื่น ให้ลองคำค้นถัดไป
            if downloaded_count < quota and attempt < max_attempts - 1:
                print(f"  [LOG] พยายามครั้งที่ {attempt+1} ไม่ครบ quota ({downloaded_count}/{quota}) ลองครั้งถัดไป...")
                await asyncio.sleep(1)  # รอสักครู่ก่อนลองใหม่
        
        # นับไฟล์จริงในโฟลเดอร์แทนการนับจากตัวแปร
        actual_count = count_actual_files_in_folder(category_folder)
        total_downloaded = actual_count
        
        # คำนวณเวลาที่ใช้ในแต่ละ query
        query_end_time = time.time()
        query_duration = query_end_time - query_start_time
        
        if downloaded_count < quota:
            print(f"\n[คำค้น: {query}] ⚠️ โหลดได้ {downloaded_count}/{quota} รูป (ไม่ครบ quota) | ใช้เวลา {query_duration:.1f} วินาที")
        else:
            print(f"\n[คำค้น: {query}] ✅ โหลดครบ quota แล้ว ({downloaded_count}/{quota}) | ใช้เวลา {query_duration:.1f} วินาที")
    
    # คำนวณเวลาทั้งหมดของหมวดหมู่
    category_end_time = time.time()
    category_duration = category_end_time - category_start_time
    print(f"[หมวด {category}] ดาวน์โหลดเสร็จสิ้น {total_downloaded} รูป | ใช้เวลาทั้งหมด {category_duration:.1f} วินาที")
    return total_downloaded

async def main():
    print("เริ่มดาวน์โหลด...")
    program_start_time = time.time()
    
    if CLEAR_FOLDER_BEFORE_RUN:
        clear_all_category_images()
    
    total_downloaded = 0
    category_times = {}
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS_BROWSER)
        page = await browser.new_page()
        async with ClientSession(timeout=ClientTimeout(total=TIMEOUT_DURATION)) as session:
            for category, search_terms in CATEGORY_SEARCHES.items():
                print(f"\n==============================\nหมวดหมู่: {category}")
                category_start_time = time.time()
                downloaded = await scrape_images_for_category(
                    category,
                    search_terms,
                    MAX_IMAGES_PER_CATEGORY,
                    MAX_RELATED_DEPTH,
                    session,
                    page
                )
                category_end_time = time.time()
                category_duration = category_end_time - category_start_time
                category_times[category] = category_duration
                total_downloaded += downloaded
                print(f"ใช้เวลา {category_duration:.1f} วินาที | ได้ {downloaded} รูป")
        await browser.close()
    
    # คำนวณเวลาทั้งหมด
    program_end_time = time.time()
    total_duration = program_end_time - program_start_time
    
    print(f"\n" + "="*60)
    print("📊 สรุปการดาวน์โหลด")
    print("="*60)
    print(f"📈 รวมทั้งหมด: {total_downloaded} รูป")
    print(f"⏱️  เวลาทั้งหมด: {total_duration:.1f} วินาที ({total_duration/60:.1f} นาที)")
    print(f"🚀 ความเร็วเฉลี่ย: {total_downloaded/total_duration:.1f} รูป/วินาที")
    
    print(f"\n📋 เวลาแต่ละหมวดหมู่:")
    for category, duration in category_times.items():
        print(f"  • {category:10}: {duration:6.1f} วินาที")
    
    print("="*60)
    
    # ทำความสะอาดไฟล์เสีย
    print(f"\n🧹 กำลังทำความสะอาดไฟล์เสีย...")
    cleaned_count = clean_corrupted_images()
    if cleaned_count > 0:
        print(f"✅ ลบไฟล์เสียแล้ว {cleaned_count} ไฟล์")
    else:
        print(f"✅ ไม่พบไฟล์เสีย")
    
    # สรุปจำนวนภาพสุดท้าย
    print(f"\n📊 สรุปจำนวนภาพสุดท้าย:")
    final_summary = get_image_summary()
    total_final = 0
    
    for category, count in final_summary.items():
        print(f"  📁 {category:10}: {count:6} รูป")
        total_final += count
    
    print(f"  📈 รวมทั้งหมด: {total_final} รูป")
    
    # เปรียบเทียบกับเป้าหมาย
    target_total = len(CATEGORY_SEARCHES) * MAX_IMAGES_PER_CATEGORY
    success_rate = (total_final / target_total) * 100
    
    print(f"\n🎯 ผลการทำงาน:")
    print(f"  • เป้าหมาย: {target_total} รูป")
    print(f"  • ได้จริง: {total_final} รูป")
    print(f"  • อัตราความสำเร็จ: {success_rate:.1f}%")
    
    if success_rate >= 95:
        print(f"  🎉 ยอดเยี่ยม! โหลดได้เกือบครบเป้าหมาย")
    elif success_rate >= 80:
        print(f"  👍 ดีมาก! โหลดได้มากกว่า 80%")
    elif success_rate >= 60:
        print(f"  ✅ ผ่าน! โหลดได้มากกว่า 60%")
    else:
        print(f"  ⚠️  ควรปรับปรุง! โหลดได้น้อยกว่า 60%")
    
    print("="*60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  หยุดการทำงานโดยผู้ใช้")
    except Exception as e:
        print(f"\n\n❌ เกิดข้อผิดพลาด: {e}")
    finally:
        # ปิด event loop อย่างสมบูรณ์
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.close()
        except:
            pass