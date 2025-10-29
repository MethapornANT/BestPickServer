# ===== ADD THIS AT THE VERY TOP OF THE FILE =====
import os, json, time, threading, hashlib, pickle, random, math
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

# Flask / JWT
from flask import Flask, request, jsonify
from functools import wraps
import jwt

# SQLAlchemy
from sqlalchemy import create_engine, text as sa_text
from sqlalchemy import text as sqltext  # some parts use sqltext explicitly

# ML / Recsys deps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from surprise import SVD, Dataset, Reader

# ==================== RECOMMENDATION SYSTEM FUNCTIONS ====================


# ============================ CONFIG  =========================

# ----------------------- DATABASE / INFRASTRUCTURE ---------------------------
# เก็บ connection string + ชื่อ table/view ที่ต้องแก้เมื่อติดตั้งบนสภาพแวดล้อมใหม่
DB_URI = os.getenv("BESTPICK_DB_URI", "mysql+mysqlconnector://root:1234@localhost/bestpick")
POSTS_TABLE   = "posts"
USERS_TABLE   = "users"
LIKES_TABLE   = "likes"
EVENT_TABLE   = "user_interactions"
CONTENT_VIEW  = "contentbasedview"   # ต้องมีคอลัมน์ตาม config ด้านล่าง
FOLLOWS_TABLE = "follower_following"

# ------------------------ USER / AUTHOR FLAGS -------------------------------
# Flag ที่กำหนดพฤติกรรมการแสดงโพสต์เจ้าของเอง และการนับโพสต์ที่เขียนเป็นสัญญาณ
INCLUDE_SELF_POSTS_IN_FEED = False   # ถ้า False จะซ่อนโพสต์ที่ user เป็นเจ้าของจาก feed
USE_AUTHORED_AS_SIGNALS     = True
AUTHORED_CATEGORY_BONUS    = 1.0    # น้ำหนักเสริมสำหรับ profile category จากโพสต์ที่เขียนเอง
AUTHORED_TEXT_BONUS        = 1.0    # น้ำหนักเสริมสำหรับ text-profile จากโพสต์ที่เขียนเอง

# ------------------------ CONTENT / FEATURE COLUMNS -------------------------
# ระบุคอลัมน์ใน content-based view / table
CATEGORY_COLS = [
    "Electronics_Gadgets",
    "Furniture",
    "Outdoor_Gear",
    "Beauty_Products",
    "Accessories",
]
TEXT_COL   = "Content"         # คอลัมน์ข้อความ (ใช้ TF-IDF)
ENGAGE_COL = "PostEngagement"  # engagement raw score (จะถูก normalize)

# --------------------- INTERACTION → IMPLICIT RATING -----------------------
# แมป action เป็นน้ำหนัก implicit rating (ใช้สร้าง SVD หรือสรุปสัญญาณ)
ACTION_WEIGHT = {
    "view": 1.0,
    "like": 2.0,
    "unlike": -1.0,
    "comment": 3.0,
    "bookmark": 4.0,
    "unbookmark": -2.0,
    "share": 5.0,
}
POS_ACTIONS = {"view", "like", "comment", "bookmark", "share"}  # ถ้านับเป็น positive
NEG_ACTIONS = {"unlike", "unbookmark"}
IGNORE_ACTIONS = {"view_profile", "follow", "unfollow"}        # ไม่ใช้ใน events
VIEW_POS_MIN = 1           # กี่ view นับเป็น positive
RATING_MIN, RATING_MAX = 0.5, 5.0  # rating scale สำหรับ Surprise SVD

# -------------------- HYBRID COMPONENT WEIGHTS (single source) -------------
# (แนะนำให้ tune ที่นี่เป็นหลัก — ถ้าต้องการยกหมวดหมู่ ให้ปรับ 'category')
HYBRID_WEIGHTS = {
    "collab": 0.20,     # collaborative SVD weight
    "item": 0.20,       # item-content / neighbor score weight
    "user_text": 0.20,  # user text-profile similarity weight
    "category": 0.40,   # category-match weight (เพิ่มค่านี้เพื่อยกหมวด)
    "pop": 0.10,        # popularity prior weight
}
# Backward-compatible globals (ฟังก์ชันเก่าเรียกชื่อเหล่านี้)
WEIGHT_COLLAB    = float(HYBRID_WEIGHTS["collab"])
WEIGHT_ITEM      = float(HYBRID_WEIGHTS["item"])
WEIGHT_USER_TEXT = float(HYBRID_WEIGHTS["user_text"])
WEIGHT_CATEGORY  = float(HYBRID_WEIGHTS["category"])
WEIGHT_POP       = float(HYBRID_WEIGHTS["pop"])

# Toggle / mapping สำหรับ integration กับ _rank
USE_HYBRID = True             # ถ้า True จะพยายามคำนวณ hybrid และใช้เป็น base_score ใน _rank
MAP_HYBRID_TO_RANK = True     # ถ้า True จะแมป category/text จาก HYBRID -> WEIGHT_C / WEIGHT_T

# -------------------- _rank-level WEIGHTS (engage/follow/recency) -----------
# _rank ใช้สัญญาณระดับโพสต์เพิ่มเติม (แยกจาก HYBRID)
WEIGHT_E = 0.20   # engagement (quality/viral)
WEIGHT_C = 0.40   # category match (จะถูก override เมื่อ MAP_HYBRID_TO_RANK=True)
WEIGHT_F = 0.15   # follow-influence category
WEIGHT_T = 0.20   # text relevance (จะถูก override เมื่อนำ HYBRID มาแมป)
WEIGHT_R = 0.05   # recency (ใช้เฉพาะ zone new / 21-30)

# ---------------- TF-IDF / ITEM-CONTENT / KNN params -----------------------
TFIDF_PARAMS = dict(
    analyzer="char_wb",
    ngram_range=(2, 5),
    max_features=60000,
    min_df=2,
    max_df=0.95,
)
KNN_NEIGHBORS = 20  # จำนวน neighbor ที่ใช้ใน KNN

# ---------------------- POPULARITY / SMOOTHING -----------------------------
POP_ALPHA = 5.0  # Bayesian smoothing สำหรับ PopularityPrior (ช่วยโพสต์ใหม่ไม่โดนลดจนน่าเกลียด)

# --------------------- FAIRNESS / CATEGORY BALANCE --------------------------
# ทำงานใน Top-K ก่อน (เช่น 20 รายการแรก) แล้วจึงปล่อยส่วน tail
FAIRNESS_TOPK       = 20     # พิจารณาความสมดุลสัดส่วนใน Top-K
FAIRNESS_RATIO_CAP  = 2.0    # สัดส่วนหมวดหลัก : หมวดรอง ไม่ควรเกิน 2:1 ใน Top-K
FAIRNESS_ALPHA      = 0.22   # ความแรงในการดึงหมวดรองขึ้นมา (0..1) สูงขึ้น = ยอม swap บ่อยขึ้น

# --- USER-SPECIFIC BALANCED PAIR OVERRIDES (Electronics ↔ Beauty 1:1) ---
# ถ้าผู้ใช้คนไหนต้องการบังคับสลับ 1:1 ใส่ user_id ลง dict นี้
BALANCED_PAIR_USERS: Dict[int, Tuple[str, str]] = {
    # ตัวอย่าง: 9999: ("Electronics_Gadgets", "Beauty_Products"),
}
# ค่าเริ่มต้นของโหมด balanced pair (เมื่อทริกเกอร์)
BALANCED_PAIR_CAP_TOP   = 2     # หัวตาราง Top-K ห้ามติดกันเกิน 1 => บังคับสลับ
BALANCED_PAIR_RATIO_CAP = 1.3   # สัดส่วน 1:1
BALANCED_PAIR_ALPHA     = 0.28  # บูสต์หมวดรองขึ้นเล็กน้อย
# ชื่อหมวดที่ถือว่าเป็น “คู่” (โหมดออโต้)
BALANCED_PAIR_NAMES = {"Electronics_Gadgets", "Beauty_Products"}

# ---------------- Cache / impression TTL / directories ---------------------
OUT_DIR = "./LogRec"
CACHE_DIR = os.path.join(OUT_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

CACHE_EXPIRY_TIME_SECONDS = 120          # อายุแคช per-user
IMPRESSION_HISTORY_TTL_SECONDS = 24*3600 # TTL ของ impression history (24 ชั่วโมง)
IMPRESSION_HISTORY_MAX_ENTRIES = 500     # เก็บ impressions สูงสุดต่อ user

# deprecated / compatibility flag (ใช้ INCLUDE_SELF_POSTS_IN_FEED แทน)
INCLUDE_SELF_POSTS = False

# ---------------- DIVERSITY / NEWNESS / THRESHOLDS -------------------------
RUNLEN_CAP_TOP20 = 3     # จำกัด run-length หมวดเดียวใน Top20
RUNLEN_CAP_AFTER = 3     # limit หลัง Top20
MMR_LAMBDA = 0.85        # MMR lambda (ใกล้ 1 => เน้น relevance)
MMR_MAX_REF = 30         # จำนวน ref item เพื่อคำนวณ diversity penalty

NEW_WINDOWS_HOURS = [1, 3, 24]  # หน้าต่างเวลา (ชั่วโมง) สำหรับนิยาม "โพสต์ใหม่"
NEW_INSERT_MAX = 3               # จำนวนโพสต์ใหม่สูงสุดที่จะแทรกใน zone 21–30

CAT_MATCH_TOP20 = 0.40   # threshold ของ category-sim สำหรับ Top20
CAT_MATCH_AFTER = 0.50   # threshold หลัง Top20
ENG_PCTL_TOP20 = 25      # percentile ของ engagement สำหรับ Top20 (e.g., 40th)
ENG_PCTL_NEW = 25        # percentile สำหรับโพสต์ใหม่

# randomization temps (biased shuffle)
TEMP_UNSEEN = 0.15
TEMP_SEENNO = 0.12
TEMP_INTER = 0.10

# ---------------- global caches / lazy artifacts (internal state) ----------
recommendation_cache: Dict[int, Dict] = {}
impression_history_cache: Dict[int, List[Dict]] = {}

# NEW: blocklist สำหรับ “โพสต์ใหม่ที่ถูกเห็นแล้ว” ต่อ user (อย่าแทรกอีก)
new_injected_seen_blocklist: Dict[int, set] = {}

_cache_lock = threading.Lock()


# lazy-built content-based artifacts (อาจถูกเติมโดย background builder)
_tfidf = None
_X = None
_postidx: Dict[int, int] = {}

# ---------------- TTL MODE DEFAULT ----------------------------------------
USE_TTL_SEEN = True  # True: ใช้ TTL-based seen (impression cache); False: ใช้ event-based seen

# ---------------- optional monitoring / safe defaults ----------------------
SEEN_ACCESS_SUMMARY_ON_RECOMMEND = False
def _seen_pop_count() -> int:
    # stub: คืนค่า 0 ถ้าไม่มีโค้ดตรวจสรุปแยกไว้
    return 0

# ---------------------- CODE AUTORELOAD / VERSION HASH ----------------------
# อย่าเรียก _as_bool ตรงนี้ เพราะยังไม่ได้ประกาศ ฟัดเอาจาก env ตรงๆ เลย
RECSYS_AUTORELOAD = str(os.getenv("RECSYS_AUTORELOAD", "true")).strip().lower() in (
    "1", "true", "t", "yes", "y", "on"
)

# รายชื่อไฟล์โค้ดที่เฝ้าดู (คอมมาแยกไฟล์); ดีฟอลต์คือไฟล์นี้เอง
RECSYS_CODE_FILES = os.getenv("RECSYS_CODE_FILES", __file__)

def _compute_code_hash() -> str:
    """
    คืนค่า MD5 ของไฟล์โค้ดทั้งหมดใน RECSYS_CODE_FILES
    ใช้ทั้งเนื้อไฟล์และ mtime รวมในแฮช เพื่อให้เปลี่ยนทันทีเมื่อแก้
    """
    h = hashlib.md5()
    for p in [x.strip() for x in str(RECSYS_CODE_FILES).split(",") if x.strip()]:
        try:
            with open(p, "rb") as f:
                h.update(f.read())
            try:
                m = os.path.getmtime(p)
                h.update(str(m).encode())
            except Exception:
                pass
        except Exception:
            # ไฟล์เปิดไม่ได้ก็ข้ามไป
            continue
    return h.hexdigest()

CODE_VERSION_HASH = _compute_code_hash()

# -------------------- SEEN / REFRESH PENALTY SETTINGS --------------------
# โพสต์ที่เพิ่งถูกเห็นภายในเวลานี้ จะถูก "cooldown" ไม่ให้โผล่ด้านบน
NO_SHOW_COOLDOWN_SECONDS = 600    # 10 นาทีแรก ถ้าไม่มี interaction ให้ถอยไปท้ายเลย
# น้ำหนักลงโทษสกอร์ตามเวลา (ยิ่งเพิ่งเห็น ยิ่งโดนหนัก)
SEEN_PENALTY_ALPHA = 0.95         # 0.95 = ลดสกอร์ลงสูงสุด ~95% เมื่อเพิ่งเห็นสดๆ
SEEN_HALF_LIFE_SECONDS = 3600     # ครึ่งชีวิต 1 ชม. (โทษจะค่อยๆ จางตามเวลา)

# --------------------------- SHORT GUIDANCE --------------------------------
# - ถ้าจะ "ยกหมวดหมู่" ให้ปรับ HYBRID_WEIGHTS['category'] (หรือ WEIGHT_CATEGORY ถาจะไม่ใช้ dict)
# - ถ้าต้องการ tune ให้เป็นเอกภาพ: ปรับ HYBRID_WEIGHTS แล้วตั้ง MAP_HYBRID_TO_RANK=True
# - ปรับ USE_HYBRID=True เฉพาะเมื่อเตรียม artifacts (tfidf/knn/svd) ไว้แล้วหรืออนุญาตให้สร้าง background
# ============================================================================ 

# ================================ UTILITIES =====================================

from sqlalchemy import text as sqltext

def _get_authored_ids(e, user_id: int) -> List[int]:
    """ดึง id ของโพสต์ที่ user เป็นเจ้าของ (สถานะ active ถ้ามี)"""
    try:
        df = pd.read_sql(
            sqltext("SELECT id FROM posts WHERE user_id = :uid AND (status='active' OR status IS NULL)"),
            e, params={"uid": int(user_id)}
        )
        return pd.to_numeric(df["id"], errors="coerce").dropna().astype(int).tolist()
    except Exception:
        return []

def _normalize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce').fillna(0.0).astype(np.float32)
    mn, mx = float(s.min()), float(s.max())
    return (s - mn) / (mx - mn + 1e-12)

def _md5_of_df(df: pd.DataFrame, cols: List[str]) -> str:
    """
    Robust MD5 snapshot of selected columns.
    Uses pandas.hash_pandas_object and converts to bytes safely.
    """
    snap = df[cols].copy().fillna(0)
    try:
        arr = pd.util.hash_pandas_object(snap, index=False).values
        # ensure bytes (works for numpy dtypes)
        b = arr.tobytes()
    except Exception:
        # fallback to deterministic JSON bytes (slower but safe)
        try:
            b = json.dumps(snap.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        except Exception:
            # ultimate fallback
            b = str(snap.values.tolist()).encode("utf-8")
    return hashlib.md5(b).hexdigest()

def _atomic_write_file(path: str, data_bytes: bytes):
    """Write bytes atomically (write tmp -> replace)."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(data_bytes)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp, path)
    except Exception:
        # best-effort; if atomic replace fails, try simple write
        try:
            with open(path, "wb") as f:
                f.write(data_bytes)
        except Exception:
            pass

# เพิ่มไว้บนหัวไฟล์
_ENGINE = None

def _eng():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(DB_URI, pool_pre_ping=True, pool_recycle=1800)
    return _ENGINE

def _flush_caches_and_restart(reason: str = "code-changed"):
    try:
        _append_rec_log([f"[{_fmt_th(_now_th())}][autoreload] restarting due to {reason}"])
    except Exception:
        pass

    _HEALTH["reloading"] = True
    try:
        with _cache_lock:
            recommendation_cache.clear()
            impression_history_cache.clear()
            new_injected_seen_blocklist.clear()
    except Exception:
        pass

    # ปิด DB pool ให้เรียบร้อยก่อนรีสตาร์ต
    try:
        global _ENGINE
        if _ENGINE is not None:
            _ENGINE.dispose()
            _ENGINE = None
    except Exception:
        pass

    try:
        import sys
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception:
        os._exit(121)

# เพิ่มบนหัวไฟล์
import py_compile

_HEALTH = {"reloading": False}
_WATCHER_STARTED = False  # กันสตาร์ตซ้ำ
_DEBOUNCE_SEC = 0.8       # รอไฟล์นิ่งก่อนคอมไพล์/รีสตาร์ต

def _list_code_files() -> list:
    """รวมไฟล์ .py ทั้งโปรเจ็กต์ (ยกเว้น venv/__pycache__/.git) หรือใช้ RECSYS_CODE_FILES ถ้ามี"""
    roots = [os.getenv("RECSYS_CODE_ROOT", os.path.dirname(__file__))]
    files = []
    allow = tuple(".py",)
    deny_dirs = {"__pycache__", ".git", "venv", ".venv", "env", ".idea", ".mypy_cache", ".pytest_cache"}
    custom = os.getenv("RECSYS_CODE_FILES")
    if custom:
        return [x.strip() for x in custom.split(",") if x.strip()]
    for root in roots:
        for d, subdirs, fns in os.walk(root):
            base = os.path.basename(d)
            if base in deny_dirs:
                continue
            for fn in fns:
                if fn.endswith(".py"):
                    files.append(os.path.join(d, fn))
    # ถ้าไม่มีอะไรจริงๆ ให้ fallback เป็นไฟล์นี้
    return files or [__file__]

def _code_is_compilable(paths: list) -> (bool, str):
    """คอมไพล์ไฟล์ทั้งหมดแบบ dry-run; ถ้าพัง ส่งข้อความ error กลับมา"""
    try:
        for p in paths:
            try:
                py_compile.compile(p, doraise=True)
            except py_compile.PyCompileError as ex:
                return False, f"{p}: {ex.msg}"
        return True, ""
    except Exception as ex:
        return False, str(ex)

def _start_code_change_watcher(poll_seconds: float = 1.5):
    global _WATCHER_STARTED
    if _WATCHER_STARTED or not RECSYS_AUTORELOAD:
        return
    _WATCHER_STARTED = True

    files = _list_code_files()
    mtimes = {}
    for p in files:
        try:
            mtimes[p] = os.path.getmtime(p)
        except Exception:
            mtimes[p] = None

    def _loop():
        last_change = 0.0
        while True:
            try:
                changed = False
                # รีสแกนไฟล์เป็นพักๆ (กันไฟล์ใหม่)
                scan_files = _list_code_files()
                for p in scan_files:
                    try:
                        m = os.path.getmtime(p)
                    except Exception:
                        m = None
                    if mtimes.get(p) != m:
                        mtimes[p] = m
                        changed = True
                        last_change = time.time()
                if changed:
                    # debounce: รอให้ไฟล์นิ่ง
                    while time.time() - last_change < _DEBOUNCE_SEC:
                        time.sleep(0.1)

                    ok, msg = _code_is_compilable(scan_files)
                    if not ok:
                        _append_rec_log([f"[{_fmt_th(_now_th())}][autoreload][SKIP] syntax error: {msg}"])
                        # อย่าตาย! รอแก้แล้วค่อยลองใหม่
                    else:
                        _HEALTH["reloading"] = True
                        _append_rec_log([f"[{_fmt_th(_now_th())}][autoreload] restart (code clean)"])
                        _flush_caches_and_restart("code-file-updated")
                        return
            except Exception as ex:
                _append_rec_log([f"[{_fmt_th(_now_th())}][autoreload][WARN] {ex}"])
            time.sleep(max(0.5, float(poll_seconds)))

    try:
        t = threading.Thread(target=_loop, daemon=True)
        t.start()
    except Exception:
        pass

# เรียก watcher ตอนโหลดโมดูล
try:
    _start_code_change_watcher()
except Exception:
    pass

# -------------------------- READ ALL LOG FILES ---------------------------------

# --- ADD: warm impressions from ALL log files (persist across restarts) ---
_impressions_warmed_from_logs = False

def _warm_impressions_from_logs(force: bool = False):
    """
    อ่านทุกไฟล์ logrec_*.txt แล้วเติม impression + blocklist ต่อ user
    ใช้เฉพาะรายการภายใน TTL ล่าสุด (IMPRESSION_HISTORY_TTL_SECONDS)
    """
    global _impressions_warmed_from_logs
    if _impressions_warmed_from_logs and not force:
        return

    try:
        import re
        from zoneinfo import ZoneInfo
        from datetime import timezone

        lines = read_all_rec_logs(OUT_DIR)
        if not lines:
            _impressions_warmed_from_logs = True
            return

        # [YYYY-mm-dd HH:MM:SS ICT][seen] uid=123 seen_ids=[1,2,3]
        rx = re.compile(
            r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) [A-Z]+\]\[seen\]\s+uid=(?P<uid>\d+)\s+seen_ids=\[(?P<ids>[0-9,\s]+)\]"
        )

        now_utc = datetime.utcnow()
        cutoff_utc = now_utc - timedelta(seconds=float(IMPRESSION_HISTORY_TTL_SECONDS))

        # เก็บชั่วคราวก่อน merge เข้า cache จริง
        tmp_map: Dict[int, List[Dict]] = defaultdict(list)

        for ln in lines:
            m = rx.match(ln)
            if not m:
                continue
            ts_str = m.group("ts")
            uid = int(m.group("uid"))
            ids_raw = m.group("ids")

            # แปลงเวลา ICT -> UTC naive
            try:
                ts_local = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("Asia/Bangkok"))
                ts_utc = ts_local.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                continue

            if ts_utc < cutoff_utc:
                continue

            # ดึงตัวเลข post ids ทั้งหมด
            try:
                import re as _re
                pids = [_ for _ in [_re.findall(r"\d+", ids_raw)][0]]
                pids = [int(x) for x in pids]
            except Exception:
                continue

            for pid in pids:
                tmp_map[uid].append({"post_id": int(pid), "ts": ts_utc})

        if not tmp_map:
            _impressions_warmed_from_logs = True
            return

        with _cache_lock:
            for uid, new_hist in tmp_map.items():
                old = impression_history_cache.get(uid, [])
                merged = old + new_hist
                # เก็บ “ล่าสุดต่อโพสต์” ภายใน TTL
                latest: Dict[int, datetime] = {}
                for h in merged:
                    pid = int(h["post_id"])
                    t = h["ts"]
                    if not isinstance(t, datetime):
                        continue
                    if t >= cutoff_utc and (pid not in latest or t > latest[pid]):
                        latest[pid] = t
                # rebuild list + sort
                merged_list = [{"post_id": pid, "ts": t} for pid, t in latest.items()]
                merged_list.sort(key=lambda x: x["ts"])
                impression_history_cache[uid] = merged_list[-IMPRESSION_HISTORY_MAX_ENTRIES:]

                # อัปเดต blocklist สำหรับการ inject (เพื่อไม่เอามา privileged อีก)
                s = new_injected_seen_blocklist.get(uid)
                if s is None:
                    s = set()
                    new_injected_seen_blocklist[uid] = s
                for h in merged_list:
                    s.add(int(h["post_id"]))

        _impressions_warmed_from_logs = True

    except Exception:
        # อย่าทำให้ทั้งระบบล้ม ถ้า parse log พัง
        _impressions_warmed_from_logs = True
        return

def iter_all_rec_logs(out_dir: str = OUT_DIR):
    """
    yield บรรทัดจากไฟล์ logrec_*.txt ทั้งหมด เรียงตามเวลาไฟล์
    """
    try:
        import glob
        paths = sorted(glob.glob(os.path.join(out_dir, "logrec_*.txt")))
        for fp in paths:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        yield line.rstrip("\n")
            except Exception:
                continue
    except Exception:
        return

def read_all_rec_logs(out_dir: str = OUT_DIR) -> List[str]:
    """
    คืน list ของทุกบรรทัดใน log ทั้งหมด
    """
    return list(iter_all_rec_logs(out_dir))

def _safe_pickle_load(path: str):
    """Return loaded object or None. If file corrupt, remove it and return None."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        try:
            os.remove(path)
        except Exception:
            pass
        return None

def _safe_get_body():
    body = request.get_json(silent=True)
    if isinstance(body, dict):
        return body
    try:
        raw = (request.data or b"").decode("utf-8", "ignore").strip()
        if raw and raw[0] in "{[":
            return json.loads(raw) or {}
    except Exception:
        pass
    if request.form:
        return {k: request.form.get(k) for k in request.form.keys()}
    if request.args:
        return {k: request.args.get(k) for k in request.args.keys()}
    return {}

def _as_bool(v, default=False):
    if isinstance(v, bool): return v
    if v is None: return default
    s = str(v).strip().lower()
    if s in ("1","true","t","yes","y","on"):  return True
    if s in ("0","false","f","no","n","off"): return False
    return default

def _as_int(v, default=0):
    try: return int(v)
    except Exception: return default

def _append_rec_log(lines: List[str], fp: Optional[str] = None):
    """
    เขียน log ลงไฟล์; ถ้าไม่ระบุ fp:
      - รอบแรกของการเรียกในโปรเซสจะสร้างไฟล์ใหม่ชื่อ logrec_<APP_START_UTC>.txt
      - รอบถัด ๆ ไปของโปรเซสเดียวกันจะ append ไฟล์เดิม
    """
    try:
        if not hasattr(_append_rec_log, "_session_fp"):
            start_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            os.makedirs(OUT_DIR, exist_ok=True)
            _append_rec_log._session_fp = os.path.join(OUT_DIR, f"logrec_{start_ts}.txt")
            # เขียน header เปิดไฟล์รอบนี้
            with open(_append_rec_log._session_fp, "a", encoding="utf-8") as f:
                f.write(f"===== recsys session started at {start_ts} =====\n")

        path = fp or getattr(_append_rec_log, "_session_fp")
        with open(path, "a", encoding="utf-8") as f:
            for ln in lines:
                f.write(ln.rstrip("\n") + "\n")
    except Exception:
        pass

# =============== STORY-LOGGING HELPERS (ADD THIS BLOCK) ===============
LOG_ENABLE_STORY = True
LOG_TOPN_PER_ITEM = int(os.getenv("LOG_TOPN_PER_ITEM", "30"))

# เข้ารหัสหมวดเป็นตัวเลข เพื่อทำ sequence ง่ายเวลาพรีเซนต์
CAT_CODE = {c: i + 1 for i, c in enumerate(CATEGORY_COLS)}
CAT_LEGEND = {i: c for c, i in CAT_CODE.items()}

def _log_human_block(title: str, sections: List[Tuple[str, List[str]]]):
    """
    เขียน human-readable summary โดยคั่นหัวข้อด้วย ////////////////
    sections: [(หัวข้อ, [รายการบรรทัด]), ...]
    """
    try:
        ts = _fmt_th(_now_th())
        lines = []
        lines.append(f"[{ts}][human] //////////////////////////////// {title} ////////////////////////////////")
        for head, items in sections:
            lines.append(f"[{ts}][human] == {head} ==")
            for s in items:
                lines.append(f"[{ts}][human] - {s}")
            lines.append(f"[{ts}][human] ------------------------------------------------------------")
        lines.append(f"[{ts}][human] //////////////////////////////////////////////////////////////////////")
        _append_rec_log(lines)
    except Exception:
        pass


def _should_force_balanced_pair(user_id: int, user_events: pd.DataFrame, content_df: pd.DataFrame) -> Tuple[bool, Tuple[str, str]]:
    """
    เปิดโหมด Balanced Pair เฉพาะเมื่อ 'กำหนดไว้แบบเจาะจง user' เท่านั้น
    (ปิด auto-detect เพื่อไม่ให้เกิด ABAB 1:1 โดยไม่ได้ตั้งใจ)
    """
    pair = BALANCED_PAIR_USERS.get(int(user_id))
    if pair and len(pair) == 2:
        # primary, secondary
        return True, (str(pair[0]), str(pair[1]))
    return False, ("", "")

def _cat_code(cat: str) -> int:
    return int(CAT_CODE.get(cat, 0))

def _log_story(tag: str, payload: dict):
    """เขียน log แบบ JSONL อ่านง่ายไว้เล่าเรื่อง/ดีบักภายหลัง"""
    if not LOG_ENABLE_STORY:
        return
    try:
        ts = _fmt_th(_now_th())
        _append_rec_log([f"[{ts}][story][{tag}] {json.dumps(payload, ensure_ascii=False, default=str)}"])
    except Exception:
        pass

def _encode_cat_seq(ids: List[int], content_df: pd.DataFrame) -> List[int]:
    def _cat_of(pid):
        return category_by_pid(content_df, int(pid))
    return [_cat_code(_cat_of(pid)) for pid in ids]

def _fairness_stats(order_ids: List[int], content_df: pd.DataFrame, topk: int = 20) -> Dict[str, Any]:
    """คืนค่าสถิติ fairness แบบสั้น ๆ สำหรับโชว์สไลด์"""
    seq = _encode_cat_seq(order_ids[:max(1, topk)], content_df)
    # max run-length
    max_run = 0; cur = 0; last = None
    for c in seq:
        if c == last:
            cur += 1
        else:
            cur = 1; last = c
        if cur > max_run:
            max_run = cur
    # ratio ของ top2 ภายใน window
    cnt = defaultdict(int)
    for c in seq:
        cnt[c] += 1
    pairs = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
    if len(pairs) >= 2:
        a, b = pairs[0][1], pairs[1][1]
        ratio = float(a) / float(max(1, b))
        top2_codes = (pairs[0][0], pairs[1][0])
    elif len(pairs) == 1:
        ratio = float("inf"); top2_codes = (pairs[0][0], None)
    else:
        ratio = 0.0; top2_codes = (None, None)
    return {
        "max_run_topk": int(max_run),
        "ratio_top2_topk": float(ratio),
        "top2_codes": top2_codes,
        "legend": CAT_LEGEND
    }
# =====================================================================

def _log_recommendation(uid: int, start: int, page_size: int, return_all: bool, posts: List[dict]):
    try:
        ts = _fmt_th(_now_th())
        ids = [int(p["id"]) for p in posts]
        cats = []
        try:
            e = _eng(); content_df = _load_content_view(e)
            idx = content_df.set_index("post_id")
            # ADD: log ขนาด content base
            _append_rec_log([f"[{ts}][content_base] total_post_ids={len(idx.index)}"])
            for p in posts:
                pid = int(p["id"])
                if pid in idx.index:
                    vals = idx.loc[pid, CATEGORY_COLS].to_numpy(dtype=np.float32)
                    cat = CATEGORY_COLS[int(np.argmax(vals))] if vals.size else "Unknown"
                else:
                    cat = "Unknown"
                cats.append(cat)
        except Exception:
            cats = ["Unknown"] * len(posts)

        lines = []
        lines.append(f"[{ts}][recommend/posts] uid={uid} start={start} size={page_size if not return_all else len(posts)} returned={len(posts)}")
        seg_line = " | ".join(f"{i+1}:{ids[i]}:{cats[i]}" for i in range(len(posts)))
        lo = start+1 if not return_all else 1
        hi = start+len(posts) if not return_all else len(posts)
        lines.append(f"[{ts}][segments][uid={uid}][{lo}-{hi}] {seg_line}")
        lines.append(f"[{ts}][order] ids={ids}")
        lines.append(f"[{ts}][order] categories={cats}")
        _append_rec_log(lines)
    except Exception:
        pass


# ================================ SECURITY ======================================
def verify_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "No token provided or incorrect format"}), 403
        token = auth_header.split(" ")[1]
        try:
            decoded = jwt.decode(token, os.getenv("JWT_SECRET", "changeme"), algorithms=["HS256"])
            request.user_id = decoded.get("id")
            request.role = decoded.get("role")
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Unauthorized: Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Unauthorized: Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated_function

# ============================ DATA LOADING / PREP ===============================

def _load_content_view(e) -> pd.DataFrame:
    """โหลดฟีเจอร์จาก content view แล้วเตรียมคอลัมน์ที่จำเป็นทั้งหมด"""
    df = pd.read_sql(f"SELECT * FROM {CONTENT_VIEW}", e)
    if "post_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "post_id"})
    df["post_id"] = pd.to_numeric(df["post_id"], errors="coerce")
    df = df.dropna(subset=["post_id"]).copy()
    df["post_id"] = df["post_id"].astype(int)

    # ข้อความ/Engagement
    if TEXT_COL not in df.columns:   df[TEXT_COL] = ""
    if ENGAGE_COL not in df.columns: df[ENGAGE_COL] = 0.0
    eng_series = pd.to_numeric(df[ENGAGE_COL], errors="coerce").fillna(0.0).astype(np.float32)

    # Popularity prior + normalized engagement
    prior = (eng_series + POP_ALPHA) / (float(eng_series.max()) + POP_ALPHA if float(eng_series.max()) > 0 else POP_ALPHA)
    df["PopularityPrior"]     = _normalize_series(pd.Series(prior))
    df["NormalizedEngagement"] = _normalize_series(eng_series)

    # Category cols default 0
    for c in CATEGORY_COLS:
        if c not in df.columns: df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)
    return df

def _load_events_all(e) -> pd.DataFrame:
    base = "user_id, post_id, action_type"
    ev = pd.read_sql(f"SELECT {base} FROM {EVENT_TABLE}", e)
    ev["user_id"] = pd.to_numeric(ev["user_id"], errors="coerce")
    ev["post_id"] = pd.to_numeric(ev["post_id"], errors="coerce")
    ev = ev.dropna(subset=["user_id","post_id"]).copy()
    ev["user_id"] = ev["user_id"].astype(int)
    ev["post_id"] = ev["post_id"].astype(int)
    ev["action_type"] = ev["action_type"].astype(str).str.lower()
    ev = ev[~ev["action_type"].isin(IGNORE_ACTIONS)]
    return ev

# =============================== IMPRESSIONS ====================================
def _get_impressions(user_id: int) -> List[Dict]:
    # NEW: อุ่นจากทุกไฟล์ log เข้ามาในหน่วยความจำก่อน (ครั้งเดียวต่อโปรเซส)
    _warm_impressions_from_logs()

    now = datetime.utcnow()
    hist = impression_history_cache.get(user_id, [])
    # prune ตาม TTL ทุกครั้งที่อ่าน
    hist = [h for h in hist if isinstance(h.get("ts"), datetime) and (now - h["ts"]).total_seconds() < IMPRESSION_HISTORY_TTL_SECONDS]
    impression_history_cache[user_id] = hist[-IMPRESSION_HISTORY_MAX_ENTRIES:]
    return impression_history_cache[user_id]

def _record_impressions(user_id: int, post_ids: List[int]):
    now = datetime.utcnow()
    hist = _get_impressions(user_id)
    for pid in post_ids:
        hist.append({"post_id": int(pid), "ts": now})
    impression_history_cache[user_id] = hist[-IMPRESSION_HISTORY_MAX_ENTRIES:]

def _cache_janitor():
    """
    Periodically:
      - prune stale entries in recommendation_cache (by timestamp)
      - prune impression_history_cache entries older than TTL and cap per-user history
    """
    while True:
        try:
            now = datetime.utcnow()
            with _cache_lock:
                # prune recommendation_cache entries older than expiry
                stale_keys = []
                for k, v in list(recommendation_cache.items()):
                    ts = v.get("timestamp")
                    try:
                        if ts is None or (now - ts).total_seconds() >= CACHE_EXPIRY_TIME_SECONDS:
                            stale_keys.append(k)
                    except Exception:
                        stale_keys.append(k)
                for k in stale_keys:
                    recommendation_cache.pop(k, None)

                # prune impression_history_cache by TTL and cap entries per user
                for uid, hist in list(impression_history_cache.items()):
                    try:
                        newhist = [h for h in hist if (now - h["ts"]).total_seconds() < IMPRESSION_HISTORY_TTL_SECONDS]
                        if newhist:
                            impression_history_cache[uid] = newhist[-IMPRESSION_HISTORY_MAX_ENTRIES:]
                        else:
                            impression_history_cache.pop(uid, None)
                    except Exception:
                        # if structure unexpected, remove it to avoid uncontrolled growth
                        impression_history_cache.pop(uid, None)
        except Exception:
            # swallow errors to avoid terminating the janitor thread
            pass
        time.sleep(CACHE_EXPIRY_TIME_SECONDS)

# ------------------ ADD: simple file lock helpers (UTILITIES) ------------------
def _acquire_simple_lock(lock_path: str, wait_seconds: float = 30.0, poll: float = 0.5) -> bool:
    """
    Try to create a lock file atomically. If exists, wait up to wait_seconds.
    Returns True if lock acquired, False otherwise.
    Lock file content: pid + iso timestamp
    """
    start = time.time()
    pid = os.getpid()
    while True:
        try:
            # O_CREAT | O_EXCL ensures atomic create
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(f"{pid}\n{datetime.utcnow().isoformat()}Z\n")
                return True
            except Exception:
                try:
                    os.close(fd)
                except Exception:
                    pass
                # fallthrough to wait
        except FileExistsError:
            # lock exists -> check age; if stale, remove it
            try:
                stat = os.stat(lock_path)
                age = time.time() - stat.st_mtime
                # if lock older than 10 * wait_seconds, consider stale and remove
                if age > max(60.0, wait_seconds * 10):
                    try:
                        os.remove(lock_path)
                        # retry immediately
                        continue
                    except Exception:
                        pass
            except Exception:
                pass
            if (time.time() - start) >= wait_seconds:
                return False
            time.sleep(poll)
        except Exception:
            # unknown problem creating lock -> fail safe
            return False

def _release_simple_lock(lock_path: str):
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass


# ------------------ ADD: background model builder wrapper ------------------
def _ensure_models_worker(content_df: pd.DataFrame, events_all: pd.DataFrame, cache_key: str, cache_dir: str):
    """
    Worker that actually builds models and writes cache files atomically.
    This runs in a daemon thread if non-blocking mode is used.
    """
    fp_tfidf = os.path.join(cache_dir, cache_key + ".tfidf.pkl")
    fp_X     = os.path.join(cache_dir, cache_key + ".X.npz")
    fp_knn   = os.path.join(cache_dir, cache_key + ".knn.pkl")
    fp_item  = os.path.join(cache_dir, cache_key + ".item.npy")
    fp_ut    = os.path.join(cache_dir, cache_key + ".ut.pkl")
    fp_svd   = os.path.join(cache_dir, cache_key + ".svd.pkl")
    lock_path = os.path.join(cache_dir, cache_key + ".lock")

    acquired = _acquire_simple_lock(lock_path, wait_seconds=10.0)
    if not acquired:
        # someone else building or cannot get lock — give up quietly
        return
    try:
        # build TF-IDF/X if missing
        tfidf = _safe_pickle_load(fp_tfidf)
        X = None
        if tfidf is None or not os.path.exists(fp_X):
            try:
                tfidf, X, postidx, _ = build_contentbased_models(content_df)
                # atomic write
                try:
                    _atomic_write_file(fp_tfidf, pickle.dumps(tfidf))
                except Exception:
                    pass
                try:
                    save_npz(fp_X, X)
                except Exception:
                    pass
            except Exception:
                # cannot build tfidf -> abort worker
                return
        else:
            try:
                X = load_npz(fp_X).astype(np.float32)
            except Exception:
                # fallback: rebuild
                try:
                    tfidf, X, postidx, _ = build_contentbased_models(content_df)
                    _atomic_write_file(fp_tfidf, pickle.dumps(tfidf))
                    save_npz(fp_X, X)
                except Exception:
                    return

        # build knn + item_scores if missing
        knn = _safe_pickle_load(fp_knn)
        item_scores = None
        if knn is None or not os.path.exists(fp_item):
            try:
                knn = _build_knn(X)
                item_scores = _precompute_item_content_scores(knn, content_df, X)
                try:
                    _atomic_write_file(fp_knn, pickle.dumps(knn))
                except Exception:
                    pass
                try:
                    np.save(fp_item, item_scores)
                except Exception:
                    pass
            except Exception:
                # skip but don't fatal
                knn = None

        # build user text profiles (ut_profiles)
        ut = _safe_pickle_load(fp_ut)
        if ut is None:
            try:
                # create train_pos like in main flow (positive actions)
                t = events_all.groupby(["user_id","post_id","action_type"]).size().reset_index(name="cnt")
                if t.empty:
                    train_pos = pd.DataFrame(columns=["user_id","post_id"])
                else:
                    pvt = t.pivot_table(index=["user_id","post_id"], columns="action_type",
                                        values="cnt", fill_value=0, aggfunc="sum").reset_index()
                    pvt.columns = [str(c).lower() for c in pvt.columns]
                    pos = np.zeros(len(pvt), dtype=bool)
                    for a in POS_ACTIONS:
                        if a in pvt.columns: pos |= (pvt[a].to_numpy(dtype=float) > 0)
                    if "view" in pvt.columns: pos |= (pvt["view"].to_numpy(dtype=float) >= VIEW_POS_MIN)
                    if NEG_ACTIONS:
                        neg = np.zeros(len(pvt), dtype=bool)
                        for a in NEG_ACTIONS:
                            if a in pvt.columns: neg |= (pvt[a].to_numpy(dtype=float) > 0)
                        pos = np.where(neg, False, pos)
                    labels = pvt[["user_id","post_id"]].copy(); labels["y"] = pos.astype(int)
                    train_pos = labels[labels["y"]==1][["user_id","post_id"]]
                ut_profiles = _user_text_profiles(train_pos, content_df, X)
                try:
                    _atomic_write_file(fp_ut, pickle.dumps(ut_profiles))
                except Exception:
                    pass
            except Exception:
                pass

        # build collaborative SVD (with resource-safe fallback)
        svd = _safe_pickle_load(fp_svd)
        if svd is None:
            try:
                svd = build_collaborative_model(events_all, content_df["post_id"].astype(int).tolist())
                # if build_collaborative_model is heavy/failed, try a lighter attempt
            except Exception:
                svd = None

            # fallback lighter SVD attempt if failed
            if svd is None:
                try:
                    # lightweight SVD: fewer factors/epochs to avoid OOM
                    e = events_all[events_all["post_id"].isin(content_df["post_id"].astype(int).tolist())].copy()
                    if not e.empty:
                        t = e.groupby(["user_id","post_id","action_type"]).size().reset_index(name="cnt")
                        pvt = t.pivot_table(index=["user_id","post_id"], columns="action_type",
                                            values="cnt", fill_value=0, aggfunc="sum").reset_index()
                        rating = np.zeros(len(pvt), dtype=np.float32)
                        for act, w in ACTION_WEIGHT.items():
                            if act in pvt.columns:
                                rating += np.float32(w) * pvt[act].to_numpy(dtype=np.float32)
                        if "view" in pvt.columns:
                            rating += np.where(pvt["view"].to_numpy(dtype=np.float32) >= VIEW_POS_MIN, np.float32(2.0), np.float32(0.0))
                        rating = np.clip(rating, RATING_MIN, RATING_MAX)
                        data = pvt[["user_id","post_id"]].copy()
                        data["rating"] = rating
                        data = data[data["rating"] > 0]
                        if not data.empty:
                            reader = Reader(rating_scale=(RATING_MIN, RATING_MAX))
                            dset = Dataset.load_from_df(data[["user_id","post_id","rating"]], reader)
                            trainset = dset.build_full_trainset()
                            model = SVD(n_factors=64, n_epochs=10, lr_all=0.01, reg_all=0.5)
                            model.fit(trainset)
                            svd = model
                except Exception:
                    svd = None

            if svd is not None:
                try:
                    _atomic_write_file(fp_svd, pickle.dumps(svd))
                except Exception:
                    pass
    finally:
        _release_simple_lock(lock_path)


# ------------------ ADD: ensure_models_built (call this from get_hybridrecommendation_order) ------------------
def ensure_models_built(content_df: pd.DataFrame, events_all: pd.DataFrame, cache_key: str, cache_dir: str = CACHE_DIR,
                        force: bool = False, non_blocking: bool = True) -> None:
    """
    Ensure tfidf/X, knn/item_scores, ut_profiles, and svd are built (cached).
    - If non_blocking=True and models missing, spawn a daemon thread to build them and return immediately.
    - If non_blocking=False, will block trying to acquire lock and build (up to lock timeout).
    """
    fp_tfidf = os.path.join(cache_dir, cache_key + ".tfidf.pkl")
    fp_X     = os.path.join(cache_dir, cache_key + ".X.npz")
    fp_knn   = os.path.join(cache_dir, cache_key + ".knn.pkl")
    fp_item  = os.path.join(cache_dir, cache_key + ".item.npy")
    fp_ut    = os.path.join(cache_dir, cache_key + ".ut.pkl")
    fp_svd   = os.path.join(cache_dir, cache_key + ".svd.pkl")

    need_build = force or not (os.path.exists(fp_tfidf) and os.path.exists(fp_X) and os.path.exists(fp_knn) and os.path.exists(fp_item) and os.path.exists(fp_ut) and os.path.exists(fp_svd))

    if not need_build:
        return

    lock_path = os.path.join(cache_dir, cache_key + ".lock")
    if non_blocking:
        # spawn background thread to do the heavy lifting
        try:
            th = threading.Thread(target=_ensure_models_worker, args=(content_df, events_all, cache_key, cache_dir), daemon=True)
            th.start()
            return
        except Exception:
            # if cannot spawn, fallback to blocking attempt
            non_blocking = False

    # blocking path: try to acquire lock and build inline (use small timeout)
    acquired = _acquire_simple_lock(lock_path, wait_seconds=30.0)
    if not acquired:
        # cannot obtain lock => another process is building; return
        return
    try:
        # call worker inline (reuse same function)
        _ensure_models_worker(content_df, events_all, cache_key, cache_dir)
    finally:
        _release_simple_lock(lock_path)


# =========================== CONTENT-BASED (Models) =============================
def build_contentbased_models(content_df: pd.DataFrame):
    """TF-IDF + KNN + item-content score (เพื่อนบ้านเฉลี่ย Engagement)"""
    global _tfidf, _X, _postidx
    if _tfidf is not None and _X is not None and _postidx:
        return _tfidf, _X, _postidx, None  # item scoresจะคำนวณข้างล่าง
    texts = content_df[TEXT_COL].fillna("").astype(str).tolist()
    tfidf = TfidfVectorizer(**TFIDF_PARAMS, dtype=np.float32)
    X = tfidf.fit_transform(texts).astype(np.float32)
    pid_list = content_df["post_id"].astype(int).tolist()
    postidx = {pid: i for i, pid in enumerate(pid_list)}
    _tfidf, _X, _postidx = tfidf, X, postidx
    return tfidf, X, postidx, None

def _build_knn(X: csr_matrix):
    knn = NearestNeighbors(n_neighbors=KNN_NEIGHBORS, metric="cosine")
    knn.fit(X)
    return knn

def _precompute_item_content_scores(knn, content_df: pd.DataFrame, X: csr_matrix) -> np.ndarray:
    """
    Vectorized version:
      - use knn.kneighbors to get neighbor indices for every item
      - compute mean of NormalizedEngagement for neighbors using numpy indexing
    """
    n = X.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    k = min(20, n)
    try:
        dists, idxs = knn.kneighbors(X, n_neighbors=k)
        # idxs shape: (n, k)
        eng = content_df["NormalizedEngagement"].to_numpy(dtype=np.float32)
        # handle possible edge cases
        if idxs.size == 0:
            return np.zeros(n, dtype=np.float32)
        scores = np.mean(eng[idxs], axis=1).astype(np.float32)
        return scores
    except Exception:
        # fallback to safe (loop) version if knn.kneighbors fails unexpectedly
        scores = np.zeros(n, dtype=np.float32)
        try:
            dists, idxs = knn.kneighbors(X, n_neighbors=min(5, n))
            for i in range(n):
                jidx = idxs[i]
                if jidx.size:
                    scores[i] = float(np.mean(content_df["NormalizedEngagement"].to_numpy(dtype=np.float32)[jidx]))
        except Exception:
            pass
        return scores

def _user_text_profile(user_id: int, user_events: pd.DataFrame, content_df: pd.DataFrame, X: csr_matrix) -> csr_matrix:
    """
    โปรไฟล์ข้อความของผู้ใช้ (1 x n_features, csr):
      - เฉลี่ยเวกเตอร์โพสต์จาก interaction เชิงบวก
      - เติมเวกเตอร์จากโพสต์ที่ตัวเองเขียนด้วยน้ำหนัก AUTHORED_TEXT_BONUS
      - กัน np.matrix โดยบังคับเป็น ndarray เสมอ
    """
    if X is None or X.shape[0] == 0:
        return csr_matrix((1, 0), dtype=np.float32)

    pid_list = content_df["post_id"].astype(int).tolist()
    pid_to_idx = {pid: i for i, pid in enumerate(pid_list)}

    ev_idxs = []
    if not user_events.empty:
        for _, r in user_events.iterrows():
            a = str(r["action_type"]).lower()
            if a in {"view","like","comment","bookmark","share"}:
                j = pid_to_idx.get(int(r["post_id"]))
                if j is not None: ev_idxs.append(j)

    au_idxs = []
    if USE_AUTHORED_AS_SIGNALS and user_id:
        try:
            e = _eng()
            authored = _get_authored_ids(e, user_id)
            for pid in authored:
                j = pid_to_idx.get(int(pid))
                if j is not None: au_idxs.append(j)
        except Exception:
            pass

    if not ev_idxs and not au_idxs:
        return csr_matrix((1, X.shape[1]), dtype=np.float32)

    # weighted average: (sum(ev) + alpha*sum(au)) / (n_ev + alpha*n_au)
    num = (X[ev_idxs].sum(axis=0) if ev_idxs else 0)
    if au_idxs:
        num = num + AUTHORED_TEXT_BONUS * X[au_idxs].sum(axis=0)

    den = float(len(ev_idxs) + AUTHORED_TEXT_BONUS * len(au_idxs))
    mean_vec = (np.asarray(num, dtype=np.float32) / max(den, 1.0))
    if mean_vec.ndim == 1:
        mean_vec = mean_vec.reshape(1, -1)

    prof = sk_normalize(mean_vec)
    return csr_matrix(prof, dtype=np.float32)


def _user_content_score(uid: int, profiles: Dict[int, csr_matrix], X: csr_matrix, idx: int) -> float:
    prof = profiles.get(int(uid))
    if prof is None or prof.nnz == 0: return 0.0
    v = X[idx]
    num = float(v.multiply(prof).sum())
    den = (np.linalg.norm(v.data) * np.linalg.norm(prof.data)) if prof.nnz>0 and v.nnz>0 else 0.0
    return float(num/den) if den>0 else 0.0

def _user_category_profile(user_id: int, user_events: pd.DataFrame, content_df: pd.DataFrame) -> np.ndarray:
    if content_df.empty:
        return np.zeros(len(CATEGORY_COLS), dtype=np.float32)

    cat_mat = content_df.set_index("post_id")[CATEGORY_COLS].astype(np.float32)
    w = np.zeros(len(CATEGORY_COLS), dtype=np.float32)

    if not user_events.empty:
        for _, r in user_events.iterrows():
            pid = int(r["post_id"]); act = str(r["action_type"]).lower()
            if pid in cat_mat.index and act in ACTION_WEIGHT:
                w += ACTION_WEIGHT[act] * cat_mat.loc[pid].values

    if USE_AUTHORED_AS_SIGNALS and user_id:
        try:
            e = _eng()
            for pid in _get_authored_ids(e, user_id):
                if pid in cat_mat.index:
                    w += AUTHORED_CATEGORY_BONUS * cat_mat.loc[pid].values
        except Exception:
            pass

    w = np.maximum(w, 0.0)
    return _normalize_series(pd.Series(w)).to_numpy(dtype=np.float32)

# =========================== COLLABORATIVE (Model) ==============================
def build_collaborative_model(events: pd.DataFrame, post_ids: List[int]):
    """สร้าง SVD จาก implicit ratings (ตัดเฉพาะโพสต์ในปัจจุบัน)"""
    e = events[events["post_id"].isin(post_ids)].copy()
    if e.empty: return None
    t = e.groupby(["user_id","post_id","action_type"]).size().reset_index(name="cnt")
    pvt = t.pivot_table(index=["user_id","post_id"], columns="action_type",
                        values="cnt", fill_value=0, aggfunc="sum").reset_index()
    rating = np.zeros(len(pvt), dtype=np.float32)
    for act, w in ACTION_WEIGHT.items():
        if act in pvt.columns:
            rating += np.float32(w) * pvt[act].to_numpy(dtype=np.float32)
    if "view" in pvt.columns:
        rating += np.where(pvt["view"].to_numpy(dtype=np.float32) >= VIEW_POS_MIN, np.float32(2.0), np.float32(0.0))
    rating = np.clip(rating, RATING_MIN, RATING_MAX)
    data = pvt[["user_id","post_id"]].copy()
    data["rating"] = rating
    data = data[data["rating"] > 0]
    if data.empty: return None
    reader = Reader(rating_scale=(RATING_MIN, RATING_MAX))
    dset = Dataset.load_from_df(data[["user_id","post_id","rating"]], reader)
    trainset = dset.build_full_trainset()
    model = SVD(n_factors=150, n_epochs=60, lr_all=0.005, reg_all=0.5)
    model.fit(trainset)
    return model

# ========================= HYBRID RECOMMENDATION ================================
def compute_hybridrecommendation_scores(
    uid: int,
    content_df: pd.DataFrame,
    tfidf, X, postidx: Dict[int,int],
    user_text_profiles: Dict[int,csr_matrix],
    collab_model,
    item_content_scores: np.ndarray,
    user_cat_prof: np.ndarray
) -> pd.DataFrame:
    """คำนวณสกอร์ต่อโพสต์: collab + item + user_text + category + pop"""
    rows = []
    collab_default = 0.5
    cat_mat = content_df[CATEGORY_COLS].to_numpy(dtype=np.float32)
    for i, row in content_df.reset_index(drop=True).iterrows():
        pid = int(row["post_id"])
        # collab
        collab = collab_default
        if collab_model is not None:
            try:
                collab = float(collab_model.predict(int(uid), pid).est)
            except Exception:
                collab = collab_default
        # item-content (เพื่อนบ้านเฉลี่ย)
        ic = float(item_content_scores[i]) if i < len(item_content_scores) else 0.0
        # user-text cosine
        ut = _user_content_score(uid, user_text_profiles, X, i)
        # category similarity (dot / norms)
        vcat = cat_mat[i]
        da = float(np.linalg.norm(vcat)); db = float(np.linalg.norm(user_cat_prof))
        cat = float(np.dot(vcat, user_cat_prof)/(da*db+1e-12)) if da>0 and db>0 else 0.0
        # popularity prior
        pop = float(row.get("PopularityPrior", 0.0))
        final = (WEIGHT_COLLAB*collab +
                 WEIGHT_ITEM*ic +
                 WEIGHT_USER_TEXT*ut +
                 WEIGHT_CATEGORY*cat +
                 WEIGHT_POP*pop)
        rows.append((pid, collab, ic, ut, cat, pop, final))
    out = pd.DataFrame(rows, columns=["post_id","collab","item","user_text","category","pop","final"])
    out["final_norm"] = _normalize_series(out["final"])
    return out.sort_values(["final_norm","final"], ascending=[False, False])

def get_hybridrecommendation_order(uid: int, use_cache: bool=True) -> List[int]:
    now = datetime.utcnow()
    e = _eng()
    content_df = _load_content_view(e)
    events_all = _load_events_all(e)

    _log_df_info("content_df_loaded", content_df, ["post_id", TEXT_COL, ENGAGE_COL] + CATEGORY_COLS)
    _log_df_info("events_all_loaded", events_all, ["user_id", "post_id", "action_type"])
    try:
        all_ids0 = pd.to_numeric(content_df["post_id"], errors="coerce").dropna().astype(int).tolist()
    except Exception:
        all_ids0 = []
    _log_stage_counts(f"uid={uid} | all_ids_from_content", all_ids0)

    if content_df is None or content_df.empty:
        return []

    # cache key: รวม hash ของโค้ดเข้าไปด้วย
    try:
        cols_content = ["post_id", ENGAGE_COL] + list(CATEGORY_COLS)
        content_hash = _md5_of_df(content_df[cols_content], cols=cols_content)
    except Exception:
        content_hash = _md5_of_df(content_df[["post_id"]], cols=["post_id"])
    try:
        cols_events = ["user_id", "post_id", "action_type"]
        events_hash = _md5_of_df(events_all[cols_events], cols=cols_events)
    except Exception:
        events_hash = _md5_of_df(events_all[["user_id","post_id"]], cols=["user_id","post_id"])

    # >>> ใส่เวอร์ชันโค้ดเข้าไป <<<
    cache_key = f"ver={CODE_VERSION_HASH}|uid={uid}|{content_hash}|{events_hash}"

    with _cache_lock:
        cached = recommendation_cache.get(uid)
        if use_cache and cached and cached.get("key") == cache_key and \
           (now - cached.get("timestamp", now)).total_seconds() < CACHE_EXPIRY_TIME_SECONDS:
            return [int(x) for x in cached["ids"]]

    user_events = events_all[events_all["user_id"] == int(uid)] if "user_id" in events_all.columns else events_all.iloc[:0]

    if USE_TTL_SEEN:
        all_ids_all = [int(x) for x in pd.to_numeric(content_df["post_id"], errors="coerce").dropna().astype(int).tolist()]
        unseen_ttl, seen_no_ttl, interacted_ttl = _split_seen_buckets(int(uid), all_ids_all, events_all)
        hist = _get_impressions(int(uid))
        if len(hist) == 0 and len(unseen_ttl) + len(seen_no_ttl) + len(interacted_ttl) != len(all_ids_all):
            unseen, seen_no, interacted = _split_to_unseen_seenno_interacted(int(uid), content_df, events_all)
        else:
            unseen, seen_no, interacted = unseen_ttl, seen_no_ttl, interacted_ttl
    else:
        unseen, seen_no, interacted = _split_to_unseen_seenno_interacted(int(uid), content_df, events_all)

    _log_stage_counts(f"uid={uid} | unseen", unseen, {"mode": "TTL" if USE_TTL_SEEN else "events"})
    _log_stage_counts(f"uid={uid} | seen_no", seen_no)
    _log_stage_counts(f"uid={uid} | interacted", interacted)

    # HYBRID
    precomputed_scores = None
    hybrid_override_C = None
    hybrid_override_T = None
    if USE_HYBRID:
        try:
            # ใช้ cache_key ที่รวมเวอร์ชันโค้ดแล้ว -> ไฟล์ artifact จะไม่ปนกัน
            ensure_models_built(content_df, events_all, cache_key, cache_dir=CACHE_DIR, non_blocking=True)
        except Exception:
            pass
        try:
            tfidf, X, postidx, _ = build_contentbased_models(content_df)
        except Exception:
            tfidf, X, postidx = (_tfidf, _X, _postidx)
        try:
            if X is not None and getattr(X, "shape", (0,))[0] > 0:
                knn = _build_knn(X)
                item_scores = _precompute_item_content_scores(knn, content_df, X)
            else:
                item_scores = np.zeros(len(content_df), dtype=np.float32)
        except Exception:
            item_scores = np.zeros(len(content_df), dtype=np.float32)

        try:
            t = events_all.groupby(["user_id","post_id","action_type"]).size().reset_index(name="cnt")
            if t.empty:
                train_pos = pd.DataFrame(columns=["user_id","post_id"])
            else:
                pvt = t.pivot_table(index=["user_id","post_id"], columns="action_type",
                                    values="cnt", fill_value=0, aggfunc="sum").reset_index()
                pvt.columns = [str(c).lower() for c in pvt.columns]
                pos = np.zeros(len(pvt), dtype=bool)
                for a in POS_ACTIONS:
                    if a in pvt.columns: pos |= (pvt[a].to_numpy(dtype=float) > 0)
                if "view" in pvt.columns: pos |= (pvt["view"].to_numpy(dtype=float) >= VIEW_POS_MIN)
                if NEG_ACTIONS:
                    neg = np.zeros(len(pvt), dtype=bool)
                    for a in NEG_ACTIONS:
                        if a in pvt.columns: neg |= (pvt[a].to_numpy(dtype=float) > 0)
                    pos = np.where(neg, False, pos)
                labels = pvt[["user_id","post_id"]].copy(); labels["y"] = pos.astype(int)
                train_pos = labels[labels["y"]==1][["user_id","post_id"]]
            user_text_profiles = _user_text_profiles(train_pos, content_df, X if X is not None else csr_matrix((0,0)))
        except Exception:
            user_text_profiles = {}

        try:
            collab_model = build_collaborative_model(events_all, content_df["post_id"].astype(int).tolist())
        except Exception:
            collab_model = None

        try:
            user_cat_prof = _user_category_profile(int(uid), user_events, content_df)
        except Exception:
            user_cat_prof = np.zeros(len(CATEGORY_COLS), dtype=np.float32)

        try:
            df_hybrid = compute_hybridrecommendation_scores(
                int(uid), content_df, tfidf, X, postidx,
                user_text_profiles, collab_model, item_scores, user_cat_prof
            )
            precomputed_scores = dict(zip(df_hybrid["post_id"].astype(int).tolist(),
                                          df_hybrid["final"].astype(float).tolist()))
            if MAP_HYBRID_TO_RANK:
                hybrid_override_C = dict(zip(df_hybrid["post_id"].astype(int), df_hybrid["category"].astype(float)))
                hybrid_override_T = dict(zip(df_hybrid["post_id"].astype(int), df_hybrid["user_text"].astype(float)))
        except Exception:
            precomputed_scores = None
            hybrid_override_C = None
            hybrid_override_T = None

    try:
        ranked = _rank(
            int(uid), content_df, user_events,
            unseen, seen_no, interacted,
            precomputed_base_score=precomputed_scores,
            hybrid_override_C=hybrid_override_C,
            hybrid_override_T=hybrid_override_T
        )
    except Exception:
        try:
            pop_series = content_df.get("PopularityPrior", None)
            if pop_series is not None:
                ordered = content_df["post_id"].astype(int).tolist()
                ordered.sort(key=lambda p: float(content_df.loc[content_df["post_id"]==p, "PopularityPrior"].values[0]) if not content_df.loc[content_df["post_id"]==p].empty else 0.0, reverse=True)
                ranked = ordered
            else:
                ranked = content_df["post_id"].astype(int).tolist()
        except Exception:
            ranked = content_df["post_id"].astype(int).tolist()

    # >>> Rebalance ก่อน cache (ตามของเดิม) <<<
    try:
        ranked = _apply_category_rebalance(
            order=ranked,
            content_df=content_df,
            user_id=int(uid),
            user_events=user_events,
            cap_top=RUNLEN_CAP_TOP20,
            cap_after=RUNLEN_CAP_AFTER,
            fairness_topk=FAIRNESS_TOPK,
            fairness_ratio_cap=FAIRNESS_RATIO_CAP,
            fairness_alpha=FAIRNESS_ALPHA
        )
    except Exception as ex:
        _append_rec_log([f"[{_fmt_th(_now_th())}][rebalance][WARN] fallback (skip) due to {ex}"])

    with _cache_lock:
        recommendation_cache[uid] = {"ids": ranked, "timestamp": now, "key": cache_key}

    return [int(x) for x in ranked]

# ======================== DB FETCH (return full post objects) ====================
def fetch_posts_by_ids(ids: List[int], user_id: int) -> List[dict]:
    if not ids:
        return []
    e = _eng()
    placeholders = ", ".join([f":id_{i}" for i in range(len(ids))])
    params = {f"id_{i}": int(pid) for i, pid in enumerate(ids)}
    params["user_id"] = int(user_id)

    sql_with_status = sqltext(f"""
        SELECT p.*, u.username, u.picture,
               (SELECT COUNT(*) FROM {LIKES_TABLE} l WHERE l.post_id = p.id AND l.user_id = :user_id) AS is_liked
        FROM {POSTS_TABLE} p
        JOIN {USERS_TABLE} u ON u.id = p.user_id
        WHERE p.status = 'active' AND p.id IN ({placeholders})
    """)
    sql_no_status = sqltext(f"""
        SELECT p.*, u.username, u.picture,
               (SELECT COUNT(*) FROM {LIKES_TABLE} l WHERE l.post_id = p.id AND l.user_id = :user_id) AS is_liked
        FROM {POSTS_TABLE} p
        JOIN {USERS_TABLE} u ON u.id = p.user_id
        WHERE p.id IN ({placeholders})
    """)

    try:
        with e.begin() as conn:
            rows = conn.execute(sql_with_status, params).mappings().all()
    except Exception:
        with e.begin() as conn:
            rows = conn.execute(sql_no_status, params).mappings().all()

    id_to_rank = {int(pid): i for i, pid in enumerate(ids)}
    rows.sort(key=lambda r: id_to_rank.get(int(r["id"]), 10**9))

    # LOG: compare requested vs fetched
    try:
        ts = _fmt_th(_now_th())
        requested_ids = [int(x) for x in ids]
        fetched_ids = [int(r["id"]) for r in rows]
        missing = [x for x in requested_ids if x not in set(fetched_ids)]
        _append_rec_log([
            f"[{ts}][fetch_posts_by_ids] requested={len(requested_ids)} fetched={len(rows)} dropped_by_sql={len(missing)} missing_sample={missing[:10]}"
        ])
    except Exception:
        pass

    out = []
    for r in rows:
        upd = r.get("updated_at") or r.get("updatedAt") or r.get("created_at") or r.get("createdAt")
        try:
            if upd is None:
                iso_updated = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
            elif isinstance(upd, str):
                dt = pd.to_datetime(upd, errors="coerce")
                iso_updated = (dt.to_pydatetime() if not pd.isna(dt) else datetime.utcnow()).replace(microsecond=0).isoformat() + "Z"
            else:
                iso_updated = upd.replace(microsecond=0).isoformat() + "Z"
        except Exception:
            iso_updated = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

        def _j(v):
            if v is None: return []
            if isinstance(v, (list, dict)): return v
            try: return json.loads(v) or []
            except Exception: return []

        out.append({
            "id": int(r["id"]),
            "userId": int(r["user_id"]),
            "title": r.get("Title"),
            "content": r.get("content") or r.get("Content"),
            "updated": iso_updated,
            "photo_url": _j(r.get("photo_url")),
            "video_url": _j(r.get("video_url")),
            "userName": r.get("username"),
            "userProfileUrl": r.get("picture"),
            "is_liked": (r.get("is_liked") or 0) > 0,
        })
    return out


# ============================ MISSING HELPERS ==================================
def _seen_penalty_map(uid: int, now: Optional[datetime] = None) -> Dict[int, float]:
    """
    คืน dict: post_id -> w (0..1) บอกความ 'สด' ของการเห็นล่าสุด
    - w ~ 1 เมื่อเพิ่งเห็นสดๆ
    - ลดลงแบบ exponential ตาม half-life (SEEN_HALF_LIFE_SECONDS)
    ใช้กับการลดสกอร์และตัดออกจาก pool ด้านบนชั่วคราว
    """
    if now is None:
        now = datetime.utcnow()
    hist = _get_impressions(uid)  # [{post_id, ts}]
    if not hist:
        return {}
    last_ts: Dict[int, datetime] = {}
    for h in hist:
        pid = int(h.get("post_id"))
        ts  = h.get("ts")
        if not isinstance(ts, datetime):
            continue
        if (pid not in last_ts) or (ts > last_ts[pid]):
            last_ts[pid] = ts

    hl = max(60.0, float(SEEN_HALF_LIFE_SECONDS))  # กันค่าผิดปกติ
    out = {}
    for pid, ts in last_ts.items():
        dt = max(0.0, (now - ts).total_seconds())
        # mapping: dt=0 => w=1.0, dt=half-life => w=0.5
        w = float(np.exp(-np.log(2.0) * dt / hl))
        out[int(pid)] = max(0.0, min(1.0, w))
    return out

def _vectorize_texts(content_df: pd.DataFrame):
    """lazy build TF-IDF/X/_postidx ให้ _rank ใช้ (แยกจาก build_contentbased_models เพื่อความเข้ากันได้)"""
    global _tfidf, _X, _postidx
    if _tfidf is not None and _X is not None and _postidx:
        return _tfidf, _X, _postidx
    texts = content_df[TEXT_COL].fillna("").astype(str).tolist()
    _tfidf = TfidfVectorizer(**TFIDF_PARAMS, dtype=np.float32)
    _X = _tfidf.fit_transform(texts).astype(np.float32)
    pids = content_df["post_id"].astype(int).tolist()
    _postidx = {pid: i for i, pid in enumerate(pids)}
    return _tfidf, _X, _postidx

def category_by_pid(content_df: pd.DataFrame, pid: int) -> str:
    row = content_df.loc[content_df["post_id"] == int(pid)]
    if row.empty:
        return "Unknown"
    vals = row.iloc[0][CATEGORY_COLS].to_numpy(dtype=np.float32)
    if vals.size == 0:
        return "Unknown"
    return CATEGORY_COLS[int(np.argmax(vals))]

def _runlen_violate(cat_seq: List[str], new_cat: str, cap: int) -> bool:
    if cap <= 0: return False
    cnt = 0
    for c in reversed(cat_seq[-10:]):
        if c == new_cat: cnt += 1
        else: break
    return cnt >= cap

def _category_counts(seq: List[int], cat_of) -> Dict[str, int]:
    cnt = defaultdict(int)
    for pid in seq:
        cnt[cat_of(pid)] += 1
    return cnt

def _find_swap_candidate(
    order: List[int],
    start_from: int,
    want_cat: str,
    forbid_cats: set,
    cat_of,
    cap_after: int,
    run_hist: List[str]
) -> Optional[int]:
    """
    หา index ใน order >= start_from ของ item ที่อยู่ในหมวด want_cat
    และการนำขึ้นไปวางจะไม่ทำให้ run-length หลัง topK เกิน cap_after
    และไม่ติดหมวดต้องห้าม (สำหรับเคสหลีกทาง)
    """
    for j in range(start_from, len(order)):
        cj = cat_of(order[j])
        if cj != want_cat or cj in forbid_cats:
            continue
        # ตรวจเบื้องต้นเรื่อง runlen ถ้าเอา item นี้ขึ้นไปต่อท้าย run_hist
        tmp_hist = run_hist[-(RUNLEN_CAP_TOP20+5):] + [cj]
        if _runlen_violate(tmp_hist, cj, cap_after):
            continue
        return j
    return None

def _rebalance_by_category(
    order_ids: List[int],
    content_df: pd.DataFrame,
    base_scores: Optional[Dict[int, float]] = None,
    *,
    top_k: int = 20,
    cap_top: int = 3,
    cap_after: int = 3,
    window: int = 20,
    ratio_cap: float = 2.0,
    fairness_alpha: float = 0.22,     # ใช้เป็นแรงดึงเข้าหา quota + บูสต์หมวดรอง
    penalty_big: float = 0.60,        # โทษหนักเมื่อจะเกิน run-length cap
    penalty_ratio: float = 0.25,      # โทษเมื่อจะทำให้สัดส่วน > ratio_cap ในหน้าต่าง
    prefer_top2_for_user: Optional[List[str]] = None,
    log_ctx: Optional[dict] = None,
) -> List[int]:
    """
    Rebalancer แบบ soft (คุม run-length + สัดส่วน + quota ภายใน Top-K) พร้อมกันแพทเทิร์น ABAB:
      - hard-avoid: ถ้าเลือกแล้ว run-length เกิน cap ปัจจุบัน จะข้ามก่อน (และยอมเฉพาะเมื่อไม่มีตัวเลือกเลย)
      - quota: คำนวณเป้าหมายสัดส่วนใน Top-K จากสัดส่วนของ pool ผสม uniform -> ไม่ดันหมวดบางหมวดไปท้ายกอง
      - urgency: ถ้าช่อง Top-K ที่เหลือน้อยกว่าจำนวนที่ยัง "ขาดโควต้า" จะได้โบนัสเร่งแทรก
      - anti-ABAB: ลดคะแนนเมื่อกำลังต่อจังหวะสลับ A-B-A-B ที่น่าเบื่อ
    """
    if not order_ids:
        return []

    # ---------- post_id -> category ----------
    pid_to_cat: Dict[int, str] = {}
    for _, r in content_df.iterrows():
        try:
            pid = int(r["post_id"])
        except Exception:
            continue
        vals = r[CATEGORY_COLS].to_numpy(dtype=np.float32)
        pid_to_cat[pid] = CATEGORY_COLS[int(np.argmax(vals))] if vals.size else "Unknown"

    # ---------- base score fallback ----------
    if base_scores is None:
        base_scores = {}
        n = len(order_ids)
        for rank, pid in enumerate(order_ids):
            base_scores[int(pid)] = float(n - rank) / max(1.0, n)

    remaining = list(order_ids)              # คิวผู้สมัคร
    out: List[int] = []                      # ผลลัพธ์
    steps_log: List[Dict[str, Any]] = []     # เก็บเหตุผลรายตำแหน่ง

    # window ในการคุมสัดส่วน/แพทเทิร์น + สถานะ run-length
    from collections import deque, defaultdict
    cat_window = deque(maxlen=max(1, int(window)))
    runlen_now = defaultdict(int)
    last_cat: Optional[str] = None

    # ---------- เตรียม "quota" เป้าหมายสำหรับ Top-K ----------
    # อิงสัดส่วนของ pool (lookahead) ผสมกับ uniform (alpha_mix) เพื่อหลีกเลี่ยง biased ไปหมวดเดียว
    look_pool = remaining[:max(60, int(top_k))]
    pool_cnt = defaultdict(int)
    pool_cats = set()
    for pid in look_pool:
        c = pid_to_cat.get(int(pid), "Unknown")
        pool_cnt[c] += 1
        pool_cats.add(c)

    C = max(1, len(pool_cats))
    N = max(1, len(look_pool))
    p_emp = {c: float(pool_cnt[c]) / float(N) for c in pool_cats}
    p_uni = {c: 1.0 / float(C) for c in pool_cats}
    alpha_mix = float(min(0.6, max(0.15, fairness_alpha)))  # ผสม uniform พอควร
    p_mix = {c: (1 - alpha_mix) * p_emp.get(c, 0.0) + alpha_mix * p_uni.get(c, 0.0) for c in pool_cats}

    # ทำ quota เป้าหมายสำหรับ Top-K (ปัดให้รวม = top_k)
    tgt = {c: int(round(p_mix[c] * int(top_k))) for c in pool_cats}
    drift = int(top_k) - sum(tgt.values())
    if drift != 0:
        # เติม/ลบที่มี p_mix สูง/ต่ำ ตามสัญญาณ drift
        order_c = sorted(pool_cats, key=lambda x: p_mix[x], reverse=(drift > 0))
        i = 0
        while drift != 0 and order_c:
            c = order_c[i % len(order_c)]
            tgt[c] += 1 if drift > 0 else -1
            drift += -1 if drift > 0 else 1
            i += 1
    # ไม่ให้มี quota ติดลบ (กัน corner case)
    for c in list(tgt.keys()):
        tgt[c] = max(0, tgt[c])

    # ตัวนับการวางจริงภายใน Top-K
    placed_topk = defaultdict(int)

    # ---------- helpers ----------
    def _cap_for_pos(pos: int) -> int:
        return int(cap_top) if pos < int(top_k) else int(cap_after)

    def _ratio_hit(cnt_map: Dict[str, int], new_c: str, ratio: float) -> bool:
        temp = cnt_map.copy()
        temp[new_c] += 1
        if not temp:
            return False
        pairs = sorted(temp.items(), key=lambda x: x[1], reverse=True)
        if len(pairs) >= 2:
            a, b = pairs[0][1], pairs[1][1]
            return (b == 0 and a > 0) or (float(a) > float(ratio) * float(max(1, b)))
        # ถ้ามีหมวดเดียวในหน้าต่างและจะยาวเกินไป ก็ถือว่าชนเล็กน้อย
        return (len(pairs) == 1 and pairs[0][1] >= max(2, _cap_for_pos(len(out))))

    def _abab_penalty(hist: deque, cand_c: str) -> float:
        # ถ้าล่าสุดเป็น A,B,A,B แล้วกำลังจะต่อ A หรือ B ให้เพนัลตี้เล็กน้อย
        if len(hist) < 4:
            return 0.0
        h = list(hist)[-4:]
        if h[0] == h[2] and h[1] == h[3] and h[0] != h[1]:
            if cand_c in (h[-1], h[-2]) and cand_c != h[-3]:
                return 1.0
        return 0.0

    # ---------- main loop ----------
    pos = 0
    while remaining:
        cap_now = _cap_for_pos(pos)
        lookahead = remaining[:60]

        # นับสัดส่วนในหน้าต่างตอนนี้ (ใช้คุม ratio_cap)
        cnt_now = defaultdict(int)
        for c in cat_window:
            cnt_now[c] += 1

        best_j = None
        best_score = -1e9
        best_dbg = {
            "pen_runlen": False,
            "pen_ratio": False,
            "pen_abab": False,
            "boost_secondary": False,
            "bonus_quota": 0.0,
            "bonus_urgency": 0.0,
            "base": 0.0,
            "score": 0.0,
            "cap_now": int(cap_now),
        }
        any_feasible = False   # มีตัวเลือกที่ไม่ชน cap run-length หรือไม่

        # ช่อง Top-K ที่เหลือ (ใช้คิด urgency)
        slots_left_topk = max(0, int(top_k) - pos)

        for j, pid in enumerate(lookahead):
            c = pid_to_cat.get(int(pid), "Unknown")
            base = float(base_scores.get(int(pid), 0.0))

            # 1) hard-avoid run-length ถ้าเกิน cap (ยกเว้นไม่มีตัวอื่นให้เลือก)
            next_run = runlen_now[c] + 1 if c == last_cat else 1
            hit_run = (cap_now > 0 and next_run > cap_now)

            # 2) ratio window penalty
            hit_ratio = _ratio_hit(cnt_now, c, float(ratio_cap))

            # 3) anti-ABAB
            pen_abab = _abab_penalty(cat_window, c)

            # 4) quota & urgency (เฉพาะในช่วง Top-K เท่านั้น)
            quota_bonus = 0.0
            urgency_bonus = 0.0
            if pos < int(top_k):
                need = int(tgt.get(c, 0)) - int(placed_topk.get(c, 0))
                if need > 0:
                    # ดึงเข้าหา quota
                    quota_bonus = float(fairness_alpha) * float(need)
                    # ถ้าใกล้หมดสลอต Top-K แต่ยังขาด → เร่งให้ขึ้นก่อน
                    if slots_left_topk > 0:
                        urgency = max(0.0, float(need) / float(slots_left_topk))
                        # scale เบาๆ กันกระโดดแรงเกิน
                        urgency_bonus = 0.5 * float(fairness_alpha) * urgency

            # 5) balanced-pair secondary boost (คงของเดิม)
            boosted = False
            if prefer_top2_for_user and len(prefer_top2_for_user) >= 2:
                if c == prefer_top2_for_user[1]:
                    quota_bonus += float(max(0.0, fairness_alpha))  # รวมกับ quota_bonus ไปเลย
                    boosted = True

            # รวมคะแนน
            score = base \
                    - (penalty_big if hit_run else 0.0) \
                    - (penalty_ratio if hit_ratio else 0.0) \
                    - (0.20 * pen_abab) \
                    + quota_bonus + urgency_bonus

            # ถือว่า "feasible" ถ้าไม่ชน run-length cap ตอนนี้
            if not hit_run:
                any_feasible = True

            # เลือกคะแนนดีที่สุด (ยังไม่ยอมของที่ชน run เว้นไม่มีตัวเลือก)
            prefer = (not hit_run) if any_feasible else True
            if best_j is None or (prefer and score > best_score) or (not any_feasible and score > best_score):
                best_j = j
                best_score = score
                best_dbg = {
                    "pen_runlen": bool(hit_run),
                    "pen_ratio": bool(hit_ratio),
                    "pen_abab": bool(pen_abab > 0.0),
                    "boost_secondary": bool(boosted),
                    "bonus_quota": float(quota_bonus),
                    "bonus_urgency": float(urgency_bonus),
                    "base": float(base),
                    "score": float(score),
                    "cap_now": int(cap_now),
                }

        # ถ้าไม่มีตัวเลือกที่ไม่ชน run-length เลย → ยอมผ่อนปรน โดยเลือกตัวที่ "ชนแต่น้อยสุด" (best_j ที่คำนวณไว้)
        if best_j is None:
            pid = remaining.pop(0)
        else:
            pid = remaining.pop(best_j)

        # อัปเดตสถานะและบันทึกเหตุผล
        c = pid_to_cat.get(int(pid), "Unknown")
        out.append(int(pid))

        if c == last_cat:
            runlen_now[c] += 1
        else:
            runlen_now = defaultdict(int)
            runlen_now[c] = 1
            last_cat = c

        cat_window.append(c)
        if pos < int(top_k):
            placed_topk[c] += 1

        try:
            steps_log.append({
                "pos": pos + 1,
                "pid": int(pid),
                "cat": c,
                "cat_code": _cat_code(c),
                **best_dbg
            })
        except Exception:
            pass

        pos += 1

    # story logs ออกไปให้ดีบั๊กต่อได้
    if isinstance(log_ctx, dict):
        log_ctx.setdefault("rebalance", {})
        log_ctx["rebalance"]["steps"] = steps_log
        log_ctx["rebalance"]["params"] = {
            "top_k": int(top_k),
            "cap_top": int(cap_top),
            "cap_after": int(cap_after),
            "window": int(window),
            "ratio_cap": float(ratio_cap),
            "alpha": float(fairness_alpha),
        }
    return out

def _apply_category_rebalance(
    order: List[int],
    content_df: pd.DataFrame,
    user_id: int,
    user_events: Optional[pd.DataFrame] = None,
    cap_top: int = RUNLEN_CAP_TOP20,
    cap_after: int = RUNLEN_CAP_AFTER,
    fairness_topk: int = FAIRNESS_TOPK,
    fairness_ratio_cap: float = FAIRNESS_RATIO_CAP,
    fairness_alpha: float = FAIRNESS_ALPHA
) -> List[int]:
    """
    ใช้ rebalancer 'เวอร์ชันใหม่' และรองรับโหมด Balanced Pair (Electronics ↔ Beauty) แบบ 1:1:
      - ถ้า user ชอบทั้ง 2 หมวด (อัตโนมัติจากโปรไฟล์ Top-2) หรือถูกระบุใน BALANCED_PAIR_USERS:
          * cap_top = 1 (บังคับสลับใน Top-K)
          * fairness_ratio_cap = 1.0
          * fairness_alpha >= BALANCED_PAIR_ALPHA
          * prefer_top2_for_user = (primary, secondary) ที่มาจากโปรไฟล์/ตั้งค่า

      - บันทึก log parameters ที่ใช้จริง

    หมายเหตุ: ถ้าไม่เข้าโหมด Balanced Pair จะใช้ค่าปกติจาก config
    """
    try:
        # เตรียม user events (ถ้ายังไม่ได้ส่งมา)
        if user_events is None:
            e = _eng()
            events_all = _load_events_all(e)
            user_events = events_all[events_all["user_id"] == int(user_id)]
        else:
            events_all = None  # ไม่จำเป็นต้องโหลดซ้ำ

        # โปรไฟล์หมวด -> Top2 เริ่มต้น (ปกติ)
        prof = _user_category_profile(int(user_id), user_events, content_df)
        if prof is not None and prof.size >= 2:
            top2_idx = np.argsort(prof)[::-1][:2]
            prefer_top2 = [CATEGORY_COLS[int(top2_idx[0])], CATEGORY_COLS[int(top2_idx[1])]]
        else:
            # fallback: นับจากหัวลิสต์เดิม
            def _cat_of(pid: int) -> str:
                return category_by_pid(content_df, int(pid))
            cnt = defaultdict(int)
            for pid in order[:max(1, int(fairness_topk))]:
                cnt[_cat_of(pid)] += 1
            top = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
            if len(top) >= 2:
                prefer_top2 = [top[0][0], top[1][0]]
            elif len(top) == 1:
                prefer_top2 = [top[0][0], top[0][0]]
            else:
                prefer_top2 = [CATEGORY_COLS[0], CATEGORY_COLS[0]]

        # ตรวจว่าจะเข้าโหมด Balanced Pair หรือไม่
        enabled_balanced, pair_from_checker = _should_force_balanced_pair(int(user_id), user_events, content_df)

        cap_top_use   = int(cap_top)
        cap_after_use = int(cap_after)
        ratio_cap_use = float(fairness_ratio_cap)
        alpha_use     = float(fairness_alpha)
        prefer_use    = prefer_top2

        if enabled_balanced:
            # บังคับค่าเฉพาะโหมด balanced
            cap_top_use   = min(cap_top_use, int(BALANCED_PAIR_CAP_TOP))
            ratio_cap_use = min(ratio_cap_use, float(BALANCED_PAIR_RATIO_CAP))
            alpha_use     = max(alpha_use, float(BALANCED_PAIR_ALPHA))
            # จัดคู่ตาม checker (จะเรียง primary=ที่ user ชอบกว่า)
            if pair_from_checker and pair_from_checker[0] and pair_from_checker[1]:
                prefer_use = [pair_from_checker[0], pair_from_checker[1]]

        # Log parameters ก่อนเรียก rebalancer
        _append_rec_log([f"[{_fmt_th(_now_th())}][rebalance] total={len(order)} "
                         f"caps=(top:{cap_top_use}, after:{cap_after_use}) top_k={fairness_topk} "
                         f"pref_two=({prefer_use[0]}, {prefer_use[1]}) "
                         f"ratio_cap={ratio_cap_use} alpha={alpha_use} "
                         f"balanced_pair={'on' if enabled_balanced else 'off'}"])

        # เรียก rebalancer
        out = _rebalance_by_category(
            order_ids=order,
            content_df=content_df,
            base_scores=None,
            top_k=int(fairness_topk),
            cap_top=int(cap_top_use),
            cap_after=int(cap_after_use),
            window=int(fairness_topk),
            ratio_cap=float(ratio_cap_use),
            fairness_alpha=float(alpha_use),
            prefer_top2_for_user=prefer_use
        )
        return out

    except Exception as ex:
        _append_rec_log([f"[{_fmt_th(_now_th())}][rebalance][WARN] fallback (skip) due to {ex}"])
        return order

def _mmr_select(candidates: List[int], scores: Dict[int,float], simfunc, lam: float, k: int, log_cb=None) -> List[int]:
    """
    เพิ่ม log_cb: callback(dict) ต่อรอบที่เลือก เช่น {"chosen":pid,"pos":n,"mmr":v,"rel":r,"div":d}
    """
    selected = []
    cand = list(dict.fromkeys(candidates))  # de-dupe preserve order
    while cand and len(selected) < k:
        best_id, best_val = None, -1e9
        best_rel, best_div = 0.0, 0.0
        for pid in cand:
            rel = scores.get(pid, 0.0)
            div = max(simfunc(pid, s) for s in selected[-MMR_MAX_REF:]) if selected else 0.0
            val = lam*rel - (1-lam)*div
            if val > best_val:
                best_val, best_id, best_rel, best_div = val, pid, rel, div
        if best_id is None:
            break
        selected.append(best_id)
        if callable(log_cb):
            try:
                log_cb({
                    "chosen": int(best_id), "pos": len(selected),
                    "mmr": float(best_val), "rel": float(best_rel), "div": float(best_div)
                })
            except Exception:
                pass
        cand = [x for x in cand if x != best_id]
    return selected

def _biased_shuffle(ids: List[int], base_scores: Dict[int,float], temp: float, seed: int) -> List[int]:
    if not ids: return []
    rnd = random.Random(seed)
    def gumbel():
        u = max(1e-12, rnd.random())
        return -math.log(-math.log(u))
    scored = [(base_scores.get(pid,0.0) + temp*gumbel(), pid) for pid in ids]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [pid for _, pid in scored]

def _recency_score(ts: Optional[pd.Timestamp], now: datetime) -> float:
    if ts is None or pd.isna(ts): return 0.0
    d = (now - ts.to_pydatetime()).total_seconds()
    if d <= 0: return 1.0
    half_life = 3*24*3600
    return float(np.exp(-np.log(2)*d/half_life))

def _text_cos(idx: int, user_prof: csr_matrix, X: csr_matrix) -> float:
    if user_prof is None or user_prof.nnz == 0 or idx < 0: return 0.0
    v = X[idx]
    num = float(v.multiply(user_prof).sum())
    den = (np.linalg.norm(v.data)*np.linalg.norm(user_prof.data)) if v.nnz>0 and user_prof.nnz>0 else 0.0
    return float(num/den) if den>0 else 0.0

def _percentile_by_category(content_df: pd.DataFrame, p: float) -> Dict[str, float]:
    out = {}
    for c in CATEGORY_COLS:
        mask = pd.to_numeric(content_df[c], errors="coerce").fillna(0.0) > 0
        vals = pd.to_numeric(content_df.loc[mask, ENGAGE_COL], errors="coerce").fillna(0.0).values
        out[c] = float(np.percentile(vals, p)) if len(vals) else 0.0
    all_vals = pd.to_numeric(content_df[ENGAGE_COL], errors="coerce").fillna(0.0).values
    out["__global__"] = float(np.percentile(all_vals, p)) if len(all_vals) else 0.0
    return out

def _category_percentiles_map(content_df: pd.DataFrame) -> Dict[str, Dict[str,float]]:
    return {
        str(ENG_PCTL_TOP20): _percentile_by_category(content_df, ENG_PCTL_TOP20),
        str(ENG_PCTL_NEW)  : _percentile_by_category(content_df, ENG_PCTL_NEW),
    }

def _is_new(ts: Optional[pd.Timestamp], now: datetime, max_hours: int) -> bool:
    if ts is None or pd.isna(ts): return False
    return (now - ts.to_pydatetime()) <= timedelta(hours=max_hours)

def _daily_seed(user_id: int) -> int:
    today = date.today().isoformat()
    return int(hashlib.md5(f"{user_id}:{today}".encode()).hexdigest()[:8], 16)

def _start_interacted_position(U: int, S: int) -> int:
    return min(max(40, U + min(S, 20)), 20 + math.ceil(0.7 * U))

def _build_scores(user_id: int,
                  content_df: pd.DataFrame,
                  user_prof: np.ndarray,
                  follow_prof: np.ndarray,
                  user_text_prof: csr_matrix,
                  now: datetime) -> Tuple[Dict[int,float], Dict[int,float], Dict[int,float], Dict[int,float], Dict[int,float]]:
    """คืน dict ของ E,C,F,T,R สำหรับทุกโพสต์ (ใช้กับ _rank)"""
    cat_mat = content_df[CATEGORY_COLS].to_numpy(dtype=np.float32)

    E_raw = pd.to_numeric(content_df[ENGAGE_COL], errors="coerce").fillna(0.0).values
    e_min, e_max = float(np.min(E_raw)), float(np.max(E_raw))
    E_norm = (E_raw - e_min) / (e_max - e_min + 1e-12)

    # หา timestamp คอลัมน์ที่เหมาะ
    ts_col = None
    for c in ["created_at","createdAt","ts","timestamp","event_time","inserted_at","updated_at","updatedAt"]:
        if c in content_df.columns:
            ts_col = c; break
    if ts_col:
        R_vec = pd.to_datetime(content_df[ts_col], errors="coerce").apply(lambda t: _recency_score(t, now)).astype(float).values
    else:
        R_vec = np.zeros(len(content_df), dtype=float)

    scores_E, scores_C, scores_F, scores_T, scores_R = {}, {}, {}, {}, {}
    for i, row in content_df.reset_index(drop=True).iterrows():
        pid = int(row["post_id"])
        v = cat_mat[i]

        # C: cosine กับโปรไฟล์หมวดของ user
        c = 0.0
        na = float(np.linalg.norm(v)); nb = float(np.linalg.norm(user_prof))
        if na>0 and nb>0:
            c = float(np.dot(v, user_prof)/(na*nb))

        # F: cosine กับโปรไฟล์หมวดที่มาจาก "คนที่ user ติดตาม"
        f = 0.0
        nb2 = float(np.linalg.norm(follow_prof))
        if na>0 and nb2>0:
            f = float(np.dot(v, follow_prof)/(na*nb2))

        # T: cosine กับโปรไฟล์ข้อความของ user
        t = _text_cos(i, user_text_prof, _X)

        scores_E[pid] = float(E_norm[i])
        scores_C[pid] = float(c)
        scores_F[pid] = float(f)
        scores_T[pid] = float(t)
        scores_R[pid] = float(R_vec[i])
    return scores_E, scores_C, scores_F, scores_T, scores_R

def _follow_category_profile(e, user_id: int, content_df: pd.DataFrame) -> np.ndarray:
    """โปรไฟล์หมวดเฉลี่ยของ 'คนที่ user นี้ติดตาม' อิงการกระทำจริงของเขา"""
    try:
        ev = pd.read_sql(
            sqltext(f"""
                SELECT ui.user_id, ui.post_id, ui.action_type
                FROM {EVENT_TABLE} ui
                INNER JOIN {FOLLOWS_TABLE} ff
                    ON ff.following_id = ui.user_id
                WHERE ff.follower_id = :uid
            """), e, params={"uid": int(user_id)}
        )
    except Exception:
        ev = pd.DataFrame(columns=["user_id","post_id","action_type"])

    if ev.empty:
        return np.zeros(len(CATEGORY_COLS), dtype=np.float32)

    ev["user_id"] = pd.to_numeric(ev["user_id"], errors="coerce")
    ev["post_id"] = pd.to_numeric(ev["post_id"], errors="coerce")
    ev = ev.dropna(subset=["user_id","post_id"]).copy()
    ev["user_id"] = ev["user_id"].astype(int)
    ev["post_id"] = ev["post_id"].astype(int)
    ev["action_type"] = ev["action_type"].astype(str).str.lower()
    ev = ev[ev["action_type"].isin(ACTION_WEIGHT.keys())]
    if ev.empty:
        return np.zeros(len(CATEGORY_COLS), dtype=np.float32)

    valid_pids = set(pd.to_numeric(content_df["post_id"], errors="coerce").dropna().astype(int).tolist())
    ev = ev[ev["post_id"].isin(valid_pids)]
    if ev.empty:
        return np.zeros(len(CATEGORY_COLS), dtype=np.float32)

    cat_mat = content_df.set_index("post_id")[CATEGORY_COLS].astype(np.float32)
    cat_mat = cat_mat.loc[cat_mat.index.intersection(valid_pids)]

    user_ids = ev["user_id"].astype(int).unique().tolist()
    uid_to_idx = {u: i for i, u in enumerate(user_ids)}
    prof = np.zeros((len(user_ids), len(CATEGORY_COLS)), dtype=np.float32)

    for _, r in ev.iterrows():
        uid_f = int(r["user_id"])
        pid = int(r["post_id"])
        act = r["action_type"]
        if pid in cat_mat.index and act in ACTION_WEIGHT and uid_f in uid_to_idx:
            prof[uid_to_idx[uid_f]] += ACTION_WEIGHT[act] * cat_mat.loc[pid].values

    if prof.size == 0:
        return np.zeros(len(CATEGORY_COLS), dtype=np.float32)

    prof = np.maximum(prof, 0.0)
    avg = prof.mean(axis=0)
    return avg / (np.linalg.norm(avg) + 1e-12)

def _user_text_profiles(train_pos: pd.DataFrame, content_df: pd.DataFrame, X: csr_matrix) -> Dict[int, csr_matrix]:
    """โปรไฟล์ข้อความ (ต่อ user) สำหรับ compute_hybridrecommendation_scores"""
    pid_to_idx = {int(pid): i for i, pid in enumerate(content_df["post_id"].astype(int).tolist())}
    profiles = {}
    if train_pos is None or train_pos.empty:
        return profiles
    for uid, g in train_pos.groupby("user_id"):
        idxs = [pid_to_idx.get(int(p)) for p in g["post_id"].tolist() if pid_to_idx.get(int(p)) is not None]
        if not idxs:
            profiles[int(uid)] = csr_matrix((1, X.shape[1]), dtype=np.float32)
            continue
        mat = X[idxs]
        mean_vec = mat.mean(axis=0)
        mean_vec = np.asarray(mean_vec, dtype=np.float32)
        if mean_vec.ndim == 1:
            mean_vec = mean_vec.reshape(1, -1)
        prof = sk_normalize(mean_vec)
        profiles[int(uid)] = csr_matrix(prof, dtype=np.float32)
    return profiles


def _last_seen_age_map(uid: int, now: Optional[datetime] = None) -> Dict[int, float]:
    """
    คืน {post_id: อายุวินาทีตั้งแต่เห็นล่าสุด} ตรงจาก impression history
    ใช้สำหรับตัดสิน 'cooldown window' โดยไม่ต้องย้อนคำนวณจาก w
    """
    if now is None:
        now = datetime.utcnow()
    hist = _get_impressions(uid)
    if not hist:
        return {}
    last_ts: Dict[int, datetime] = {}
    for h in hist:
        pid = int(h.get("post_id"))
        ts  = h.get("ts")
        if isinstance(ts, datetime):
            if (pid not in last_ts) or (ts > last_ts[pid]):
                last_ts[pid] = ts
    out = {}
    for pid, ts in last_ts.items():
        out[int(pid)] = max(0.0, (now - ts).total_seconds())
    return out


def _rank(
    user_id: int,
    content_df: pd.DataFrame,
    user_events: pd.DataFrame,
    unseen: List[int],
    seen_no: List[int],
    interacted: List[int],
    precomputed_base_score: Optional[Dict[int, float]] = None,
    hybrid_override_C: Optional[Dict[int, float]] = None,
    hybrid_override_T: Optional[Dict[int, float]] = None
) -> List[int]:
    """
    เวอร์ชันปรับใหม่ + story logs:
      - gating เหตุผล (CAT/TEXT/ENG) สำหรับ top20 candidates
      - STRICT GATE: ต้องผ่าน CAT & TEXT & ENG; ถ้าไม่มีใครผ่านครบ -> fallback ปกติ
      - BOOST: ถ้ามี strict ผ่านครบ จะบูสต์ base_score ให้กลุ่มนี้ขึ้นก่อน
      - MMR เลือก 20 ตัว พร้อม log rel/div/mmr ต่ออันดับ
      - soft re-balance (มีเหตุผลต่อ pos)
      - cooldown/seen penalty สรุปจำนวน
    """
    now = datetime.utcnow()
    TEXT_GATE = 0.40
    STRICT_GATE_BONUS = 0.28  # << บูสต์สำหรับโพสต์ที่ผ่านครบ 3 เกต

    # เตรียมเวกเตอร์/โปรไฟล์เพื่อคำนวณ E/C/F/T/R
    _vectorize_texts(content_df)
    user_prof      = _user_category_profile(user_id, user_events, content_df)
    follow_prof    = _follow_category_profile(_eng(), user_id, content_df)
    user_text_prof = _user_text_profile(user_id, user_events, content_df, _X)

    scores_E, scores_C, scores_F, scores_T, scores_R = _build_scores(
        user_id, content_df, user_prof, follow_prof, user_text_prof, now
    )

    # override C/T จาก HYBRID ถ้าเปิด
    if MAP_HYBRID_TO_RANK:
        if hybrid_override_C:
            for k, v in hybrid_override_C.items():
                scores_C[int(k)] = float(v)
        if hybrid_override_T:
            for k, v in hybrid_override_T.items():
                scores_T[int(k)] = float(v)

    # base score
    if precomputed_base_score:
        base_score = {int(pid): float(precomputed_base_score.get(int(pid), 0.0))
                      for pid in content_df["post_id"].astype(int)}
    else:
        def _final_score(E, C, F, T):
            return WEIGHT_E*E + WEIGHT_C*C + WEIGHT_F*F + WEIGHT_T*T
        base_score = {
            int(pid): _final_score(scores_E.get(pid, 0.0), scores_C.get(pid, 0.0),
                                   scores_F.get(pid, 0.0), scores_T.get(pid, 0.0))
            for pid in content_df["post_id"].astype(int)
        }

    # percentile gates
    ptiles = _category_percentiles_map(content_df)

    # self posts
    try:
        self_post_ids = set(_get_authored_ids(_eng(), user_id))
    except Exception:
        self_post_ids = set()

    # seen penalty / cooldown
    seen_w   = _seen_penalty_map(user_id, now)
    seen_age = _last_seen_age_map(user_id, now)

    pos_set = set()
    if not user_events.empty:
        am = user_events["action_type"].astype(str).str.lower()
        pos = [a.lower() for a in POS_ACTIONS] if POS_ACTIONS else []
        if pos:
            pos_set = set(pd.to_numeric(
                user_events.loc[am.isin(pos), "post_id"], errors="coerce"
            ).dropna().astype(int).tolist())

    cooldown_cut = max(30.0, float(NO_SHOW_COOLDOWN_SECONDS))
    cooldown_ids = {int(pid) for pid, age in seen_age.items()
                    if (age < cooldown_cut) and (int(pid) not in pos_set)}

    if base_score and seen_w:
        for pid in list(base_score.keys()):
            w = float(seen_w.get(int(pid), 0.0))
            if w > 0.0:
                base_score[pid] = float(base_score[pid]) * (1.0 - SEEN_PENALTY_ALPHA * w)

    # pool ตาม priority
    def _drop_forbidden(lst: List[int]) -> List[int]:
        out = []
        for pid in lst:
            if pid in cooldown_ids: continue
            if not INCLUDE_SELF_POSTS_IN_FEED and pid in self_post_ids: continue
            out.append(pid)
        return out

    pool_primary = _drop_forbidden(list(dict.fromkeys([*unseen, *seen_no])))

    # ============== GATING (CAT/TEXT/ENG) + STORY LOG ==============
    def _cat_of(pid: int) -> str:
        return category_by_pid(content_df, pid)
    def _eng_val(pid: int) -> float:
        try:
            return float(content_df.loc[content_df["post_id"] == pid, ENGAGE_COL].values[0] or 0.0)
        except Exception:
            return 0.0

    gating_logs = []
    strict_cands, or_cands = [], []  # STRICT = CAT&TEXT&ENG ; OR = (CAT or TEXT) & ENG
    thr_map = ptiles.get(str(ENG_PCTL_TOP20), {}) or {}
    for pid in pool_primary:
        c, t = scores_C.get(pid, 0.0), scores_T.get(pid, 0.0)
        cat  = _cat_of(pid)
        thr  = float(thr_map.get(cat, thr_map.get("__global__", 0.0)))
        eok  = (_eng_val(pid) >= thr)
        cat_ok = (c >= CAT_MATCH_TOP20)
        txt_ok = (t >= TEXT_GATE)

        pass_strict = bool(cat_ok and txt_ok and eok)
        pass_or     = bool((cat_ok or txt_ok) and eok)

        if pass_strict:
            strict_cands.append(pid)
        elif pass_or:
            or_cands.append(pid)

        if len(gating_logs) < LOG_TOPN_PER_ITEM:
            gating_logs.append({
                "pid": int(pid), "cat": cat, "cat_code": _cat_code(cat),
                "C": float(c), "T": float(t), "E": float(_eng_val(pid)),
                "thr_E": float(thr),
                "pass_CAT": bool(cat_ok), "pass_TEXT": bool(txt_ok), "pass_ENG": bool(eok),
                "pass_STRICT": bool(pass_strict),   # << เพิ่มฟิลด์ strict
                "passed_or": bool(pass_or),
                "base": float(base_score.get(pid, 0.0)),
            })

    gating_mode = "strict+fallback" if len(strict_cands) > 0 else "fallback-normal"
    try:
        _log_story("gating", {
            "uid": int(user_id),
            "topk": 20,
            "mode": gating_mode,
            "strict_count": len(strict_cands),
            "or_count": len(or_cands),
            "logs": gating_logs,
            "legend": CAT_LEGEND
        })
    except Exception:
        pass

    # ถ้ามี STRICT candidates -> บูสต์ + จัดให้มาก่อน; ถ้าไม่มี -> ใช้ fallback ปกติ
    if len(strict_cands) > 0:
        # บูสต์ base_score ให้กลุ่มที่ผ่านครบ 3 เกต
        for pid in strict_cands:
            base_score[int(pid)] = float(base_score.get(int(pid), 0.0)) + float(STRICT_GATE_BONUS)

        # จัดลำดับ candidate สำหรับ Top20: STRICT -> OR -> ที่เหลือ
        strict_set = set(strict_cands)
        or_set     = set(or_cands)
        rest = [p for p in pool_primary if p not in strict_set and p not in or_set]
        # ภายในแต่ละกลุ่มเรียงตาม base_score ลดหลั่น
        strict_sorted = sorted(strict_cands, key=lambda x: base_score.get(x, 0.0), reverse=True)
        or_sorted     = sorted(or_cands,     key=lambda x: base_score.get(x, 0.0), reverse=True)
        rest_sorted   = sorted(rest,         key=lambda x: base_score.get(x, 0.0), reverse=True)
        cand_for_top20 = list(dict.fromkeys([*strict_sorted, *or_sorted, *rest_sorted]))
    else:
        # Fallback ปกติ (เท่าเดิม): ใช้ OR-gated เป็นแกน ถ้าน้อยกว่า 20 ก็เติมด้วยตัวคะแนนสูง
        if len(or_cands) < 20:
            relax = sorted([p for p in pool_primary if p not in set(or_cands)],
                           key=lambda x: (scores_C.get(x, 0.0), scores_T.get(x, 0.0), base_score.get(x, 0.0)),
                           reverse=True)
            cand_for_top20 = list(dict.fromkeys([*or_cands, *relax]))
        else:
            cand_for_top20 = or_cands

    # ============== MMR (with per-step log) ==============
    mmr_logs = []
    def _mmr_cb(evt: dict):
        if len(mmr_logs) < LOG_TOPN_PER_ITEM:
            mmr_logs.append({
                "pos": int(evt.get("pos", 0)),
                "pid": int(evt.get("chosen", 0)),
                "cat": _cat_of(int(evt.get("chosen", 0))),
                "cat_code": _cat_code(_cat_of(int(evt.get("chosen", 0)))),
                "mmr": float(evt.get("mmr", 0.0)),
                "rel": float(evt.get("rel", 0.0)),
                "div": float(evt.get("div", 0.0)),
            })

    def sim(a, b):
        ia, ib = _postidx.get(a, -1), _postidx.get(b, -1)
        if ia < 0 or ib < 0: return 0.0
        va, vb = _X[ia], _X[ib]
        num = float(va.multiply(vb).sum())
        den = np.linalg.norm(va.data) * np.linalg.norm(vb.data) if va.nnz and vb.nnz else 0.0
        return num/den if den>0 else 0.0

    top20_raw = _mmr_select(cand_for_top20, base_score, sim, MMR_LAMBDA, k=20, log_cb=_mmr_cb)
    chosen = set(top20_raw)

    try:
        _log_story("mmr", {"uid": int(user_id), "lambda": float(MMR_LAMBDA), "steps": mmr_logs})
    except Exception:
        pass

    # tail (ยังไม่รวม interacted/cooldown)
    all_ids = content_df["post_id"].astype(int).tolist()
    rest_candidates = []
    for pid in all_ids:
        if pid in chosen: continue
        if pid in cooldown_ids: continue
        if not INCLUDE_SELF_POSTS_IN_FEED and pid in self_post_ids: continue
        if pid in interacted: continue
        rest_candidates.append(pid)
    tail = sorted(rest_candidates, key=lambda x: base_score.get(x, 0.0), reverse=True)

    # ---- soft re-balance (top20 + tail)
    main_before = top20_raw + tail
    rb_ctx = {}
    main_after  = _rebalance_by_category(
        order_ids=main_before, content_df=content_df, base_scores=base_score,
        cap_top=RUNLEN_CAP_TOP20, cap_after=RUNLEN_CAP_AFTER, top_k=20,
        log_ctx=rb_ctx
    )

    # ส่วน interacted/cooldown ท้าย
    interacted_tail = [pid for pid in interacted
                       if pid not in set(main_after)
                       and pid not in cooldown_ids
                       and (INCLUDE_SELF_POSTS_IN_FEED or pid not in self_post_ids)]
    interacted_tail = sorted(interacted_tail, key=lambda x: base_score.get(x, 0.0), reverse=True)

    cool_tail = [pid for pid in cooldown_ids
                 if pid not in set(main_after)
                 and (INCLUDE_SELF_POSTS_IN_FEED or pid not in self_post_ids)]
    cool_tail = sorted(cool_tail, key=lambda x: base_score.get(x, 0.0), reverse=True)

    order = main_after + interacted_tail + cool_tail

    # STORY สรุป cooldown/penalty และ fairness
    try:
        _log_story("rank.final", {
            "uid": int(user_id),
            "cooldown_count": len(cooldown_ids),
            "interacted_tail": interacted_tail[:LOG_TOPN_PER_ITEM],
            "order_head": order[:LOG_TOPN_PER_ITEM],
            "cat_seq_head": _encode_cat_seq(order[:LOG_TOPN_PER_ITEM], content_df),
            "fairness": _fairness_stats(order, content_df, topk=20),
            "rebalance_params": rb_ctx.get("rebalance", {}).get("params", {}),
            "gating_mode": gating_mode,
        })
    except Exception:
        pass

    return [int(x) for x in dict.fromkeys(order)]


def _split_to_unseen_seenno_interacted(uid: int, content_df: pd.DataFrame, events_all: pd.DataFrame):
    """
    สร้าง 3 กลุ่มสำหรับ _rank:
      - unseen: ไม่เคยเห็นเลย
      - seen_no: เคยเห็นแต่ไม่ positive
      - interacted: มี positive (ตาม POS_ACTIONS)
    หมายเหตุ: ไม่ยุ่ง created_at ตามที่มึงสั่ง
    """
    ev_u = events_all[events_all["user_id"] == int(uid)] if "user_id" in events_all.columns else events_all.iloc[:0]

    # seen = มี event อะไรก็ได้กับโพสต์นั้น
    if not ev_u.empty:
        seen_set = set(pd.to_numeric(ev_u["post_id"], errors="coerce").dropna().astype(int).tolist())
    else:
        seen_set = set()

    # interacted = positive actions เท่านั้น (ไม่ใช้ view มาเป็น positive)
    interacted_set = set()
    if not ev_u.empty and "action_type" in ev_u.columns:
        am = ev_u["action_type"].astype(str).str.lower()
        pos = [a.lower() for a in POS_ACTIONS] if POS_ACTIONS else []
        if pos:
            interacted_set = set(pd.to_numeric(ev_u.loc[am.isin(pos), "post_id"], errors="coerce").dropna().astype(int).tolist())

    all_posts = [int(x) for x in pd.to_numeric(content_df["post_id"], errors="coerce").dropna().astype(int).tolist()]

    unseen     = [pid for pid in all_posts if pid not in seen_set]
    seen_no    = [pid for pid in all_posts if (pid in seen_set and pid not in interacted_set)]
    interacted = [pid for pid in all_posts if pid in interacted_set]

    return unseen, seen_no, interacted


# ============================== ROUTE HANDLER ===================================
# หมายเหตุ: ถ้าโปรเจ็กต์คุณมี Flask app อยู่แล้ว ให้ import ฟังก์ชันนี้ไปผูก route เดิมได้
# ที่นี่สมมติคุณจะใช้ @app.route('/ai/recommend', methods=['POST'])
@verify_token
def ai_recommend_handler():
    """
    เพิ่ม story log ส่วน header เพื่อ snapshot config/weights และท้ายสุดสรุป pipeline
    """

    if _HEALTH.get("reloading"):
        return jsonify({"error": "Service is reloading, please retry in a moment."}), 503
    try:
        body = _safe_get_body()

        # user id: JWT > body > query
        uid = None
        if hasattr(request, "user_id") and request.user_id:
            uid = _as_int(request.user_id, 0)
        if not uid:
            uid = _as_int(body.get("user_id") or request.args.get("user_id"), 0)
        if not uid or uid <= 0:
            return jsonify({"error": "missing/invalid user_id"}), 400

        start      = _as_int(body.get("start") or request.args.get("start"), 0)
        page_size  = _as_int(body.get("page_size") or request.args.get("page_size"), 20)
        page_size  = max(1, min(page_size, 100))
        refresh    = _as_bool(body.get("refresh") or request.args.get("refresh"), False)

        return_all = _as_bool(body.get("all") or request.args.get("all"), True)
        debug      = _as_bool(body.get("debug") or request.args.get("debug"), False)

        if refresh:
            with _cache_lock:
                recommendation_cache.pop(uid, None)

        # ========= COLD-START SHORT-CIRCUIT =========
        def _is_brand_new_user_local(user_id: int) -> bool:
            try:
                e = _eng()
                with e.connect() as conn:
                    q_evt = conn.execute(sa_text(f"SELECT COUNT(*) FROM {EVENT_TABLE} WHERE user_id=:uid"),
                                         {"uid": int(user_id)}).scalar() or 0
                    q_like = conn.execute(sa_text(f"SELECT COUNT(*) FROM {LIKES_TABLE} WHERE user_id=:uid"),
                                          {"uid": int(user_id)}).scalar() or 0
                    q_follow = conn.execute(sa_text(
                        f"SELECT "
                        f"(SELECT COUNT(*) FROM {FOLLOWS_TABLE} WHERE follower_id=:uid) + "
                        f"(SELECT COUNT(*) FROM {FOLLOWS_TABLE} WHERE following_id=:uid)"
                    ), {"uid": int(user_id)}).scalar() or 0
                    q_authored = conn.execute(sa_text(f"SELECT COUNT(*) FROM {POSTS_TABLE} WHERE user_id=:uid"),
                                              {"uid": int(user_id)}).scalar() or 0
                return (int(q_evt) + int(q_like) + int(q_follow) + int(q_authored)) == 0
            except Exception:
                return False  # เช็คไม่ได้ให้ถือว่าไม่ใหม่ เพื่อกันพัง

        if _is_brand_new_user_local(uid):
            # ดึงคอนเทนต์ + คอลัมน์ id/eng แบบเดิม
            e = _eng()
            content_df_full = _load_content_view(e)

            id_candidates  = ["post_id", "PostID", "PostId", "POST_ID", "id", "ID"]
            eng_candidates = [ENGAGE_COL, "eng", "Engagement", "PostEngagement", "post_engagement"]

            id_col  = next((c for c in id_candidates  if c in content_df_full.columns), None)
            eng_col = next((c for c in eng_candidates if c in content_df_full.columns), None)
            if not id_col or not eng_col:
                raise RuntimeError(f"cold-start: missing id/eng column; have={list(content_df_full.columns)}")

            df = content_df_full[[id_col, eng_col]].rename(columns={id_col: "post_id", eng_col: "eng"}).copy()
            df["post_id"] = pd.to_numeric(df["post_id"], errors="coerce")
            df = df.dropna(subset=["post_id"])
            df["post_id"] = df["post_id"].astype(int)
            df["eng"]     = pd.to_numeric(df["eng"], errors="coerce").fillna(0.0)

            # ==== ใช้ระบบบัคเก็ต + penalty ที่มีอยู่แล้ว ====
            # บัคเก็ต: unseen > seen_no_pos > interacted  (อ้าง helper เดิม)
            # NOTE: ไม่เพิ่มระบบใหม่ ไม่แตะสคีมา
            events_all = _load_events_all(e)
            ordered_pool = df["post_id"].tolist()

            unseen_ids, seen_no_pos_ids, interacted_ids = _split_seen_buckets(uid, ordered_pool, events_all)

            # แผนที่โทษ/ดีเคย์สำหรับที่ถูกเห็น (อ้าง helper เดิม)
            penalty_map = _seen_penalty_map(uid, now=_now_th())

            # ฟังก์ชันเรียงในบัคเก็ต: เรียงตาม engagement และใช้ penalty สำหรับอันที่เคยเห็น/มี interaction
            def _score(pid: int) -> float:
                base = float(df.loc[df["post_id"] == pid, "eng"].values[0]) if pid in set(ordered_pool) else 0.0
                pen  = float(penalty_map.get(int(pid), 1.0))
                return base * pen

            # ภายในแต่ละบัคเก็ต: เรียงคะแนนมาก→น้อย ผูกด้วย post_id กันแกว่ง
            unseen_sorted      = sorted(unseen_ids,      key=lambda p: (_score(p), p), reverse=True)
            seen_no_pos_sorted = sorted(seen_no_pos_ids, key=lambda p: (_score(p), p), reverse=True)
            interacted_sorted  = sorted(interacted_ids,  key=lambda p: (_score(p), p), reverse=True)

            # คิวรวม: unseen → seen_no_pos → interacted
            ids_all = unseen_sorted + seen_no_pos_sorted + interacted_sorted
            total   = len(ids_all)

            # ===== fetch/page cut เดิม =====
            def _candidate_ids_for_page(ids_all: List[int], start: int, page_size: int) -> List[int]:
                if return_all:
                    seen=set(); out=[]
                    for p in ids_all:
                        if p not in seen:
                            out.append(p); seen.add(p)
                    return out
                seen=set(); out=[]
                i = max(0, start)
                target = int(page_size * (2.5 if not INCLUDE_SELF_POSTS_IN_FEED else 2.0))
                while i < len(ids_all) and len(out) < target:
                    p = ids_all[i]
                    if p not in seen:
                        out.append(p); seen.add(p)
                    i += 1
                return out

            cand_ids = _candidate_ids_for_page(ids_all, start, page_size)
            posts = fetch_posts_by_ids(cand_ids, uid)

            if not return_all:
                mp = {int(p["id"]): p for p in posts}
                page_posts, i, seen_page = [], start, set()
                while len(page_posts) < page_size and i < len(ids_all):
                    pid = int(ids_all[i])
                    if pid not in seen_page and pid in mp:
                        page_posts.append(mp[pid]); seen_page.add(pid)
                    i += 1
                posts = page_posts

            _log_recommendation(uid=uid, start=start, page_size=page_size,
                                return_all=return_all, posts=posts)

            if debug:
                return jsonify({
                    "posts": posts,
                    "debug": {
                        "mode": "cold-engagement+buckets+penalty",
                        "total_candidates": total,
                        "start": start,
                        "page_size": page_size,
                        "include_self_posts": INCLUDE_SELF_POSTS_IN_FEED,
                        "map_hybrid_to_rank": MAP_HYBRID_TO_RANK
                    }
                }), 200

            return jsonify(posts), 200
        # ========= END COLD-START SHORT-CIRCUIT =========

        # HEADER STORY: snapshot config/weights
        try:
            e = _eng()
            content_df = _load_content_view(e)
            events_all = _load_events_all(e)
            _log_story("request.header", {
                "uid": int(uid),
                "code_hash": CODE_VERSION_HASH,
                "weights": {
                    "hybrid": HYBRID_WEIGHTS,
                    "rank": {"E": WEIGHT_E, "C": WEIGHT_C, "F": WEIGHT_F, "T": WEIGHT_T, "R": WEIGHT_R},
                    "map_hybrid_to_rank": bool(MAP_HYBRID_TO_RANK)
                },
                "fairness": {
                    "topk": int(FAIRNESS_TOPK),
                    "cap_top": int(RUNLEN_CAP_TOP20),
                    "cap_after": int(RUNLEN_CAP_AFTER),
                    "ratio_cap": float(FAIRNESS_RATIO_CAP),
                    "alpha": float(FAIRNESS_ALPHA),
                },
                "inject": {
                    "positions": "random(21-30)",
                    "window_h": 24
                },
                "ttl": {
                    "use_ttl_seen": bool(USE_TTL_SEEN),
                    "history_ttl_sec": int(IMPRESSION_HISTORY_TTL_SECONDS),
                    "cooldown_sec": int(NO_SHOW_COOLDOWN_SECONDS),
                    "seen_penalty_alpha": float(SEEN_PENALTY_ALPHA),
                    "half_life_sec": int(SEEN_HALF_LIFE_SECONDS)
                },
                "data_sizes": {
                    "content_rows": int(len(content_df)),
                    "events_rows": int(len(events_all))
                },
                "legend": CAT_LEGEND
            })
        except Exception:
            pass

        # 1) อันดับหลัก
        ids_all = get_hybridrecommendation_order(uid, use_cache=(not refresh))
        total   = len(ids_all)

        # 2) inject โพสต์ใหม่เข้า slot 21–30 (สุ่มตำแหน่งอัตโนมัติ)
        ids_all = _inject_new_today(ids_all, uid, positions=None, hours=24)

        # 3) candidates สำหรับ fetch (กันดรอป inactive/self)
        def _candidate_ids_for_page(ids_all: List[int], start: int, page_size: int) -> List[int]:
            if return_all:
                seen=set(); out=[]
                for p in ids_all:
                    if p not in seen:
                        out.append(p); seen.add(p)
                return out
            # over-fetch 2.5x กันดรอป
            seen=set(); out=[]
            i = max(0, start)
            target = int(page_size * (2.5 if not INCLUDE_SELF_POSTS_IN_FEED else 2.0))
            while i < len(ids_all) and len(out) < target:
                p = ids_all[i]
                if p not in seen:
                    out.append(p); seen.add(p)
                i += 1
            return out

        cand_ids = _candidate_ids_for_page(ids_all, start, page_size)
        posts = fetch_posts_by_ids(cand_ids, uid)

        # 4) page cut
        if not return_all:
            mp = {int(p["id"]): p for p in posts}
            page_posts, i, seen_page = [], start, set()
            while len(page_posts) < page_size and i < len(ids_all):
                pid = int(ids_all[i])
                if pid not in seen_page and pid in mp:
                    page_posts.append(mp[pid]); seen_page.add(pid)
                i += 1
            posts = page_posts

        _log_recommendation(uid=uid, start=start, page_size=page_size,
                            return_all=return_all, posts=posts)

        # SUMMARY (เดิม)
        try:
            e = _eng()
            content_df_hr = _load_content_view(e)
            events_all_hr = _load_events_all(e)
            user_events_hr = events_all_hr[events_all_hr["user_id"] == int(uid)] if "user_id" in events_all_hr.columns else events_all_hr.iloc[:0]
            enabled_balanced, pair_used = _should_force_balanced_pair(int(uid), user_events_hr, content_df_hr)

            cat_seq_20 = _encode_cat_seq(ids_all[:20], content_df_hr)
            cat_seq_10 = _encode_cat_seq(ids_all[:10], content_df_hr)

            sections = [
                ("Who & Request",
                 [f"uid={uid} start={start} page_size={page_size} return_all={return_all}",
                  f"total_candidates={total} returned={len(posts)}"]),

                ("Fairness Params (final-use)",
                 [f"balanced_pair={'ON' if enabled_balanced else 'OFF'} pair={pair_used if enabled_balanced else '-'}",
                  f"top_k={FAIRNESS_TOPK} cap_top={(BALANCED_PAIR_CAP_TOP if enabled_balanced else RUNLEN_CAP_TOP20)} "
                  f"cap_after={RUNLEN_CAP_AFTER} ratio_cap={(BALANCED_PAIR_RATIO_CAP if enabled_balanced else FAIRNESS_RATIO_CAP)} "
                  f"alpha={(max(FAIRNESS_ALPHA, BALANCED_PAIR_ALPHA) if enabled_balanced else FAIRNESS_ALPHA)}"]),

                ("Top20 Category Seq (codes)",
                 [",".join(map(str, cat_seq_20))]),
                ("Top10 Category Seq (codes)",
                 [",".join(map(str, cat_seq_10))]),
                ("Top IDs (head)",
                 [",".join(map(str, ids_all[:min(LOG_TOPN_PER_ITEM, len(ids_all))]))]),
            ]
            _log_human_block("RECSYS SUMMARY", sections)
        except Exception:
            pass

        if debug:
            return jsonify({
                "posts": posts,
                "debug": {
                    "total_candidates": total,
                    "start": start,
                    "page_size": page_size,
                    "weights": {
                        "collab": WEIGHT_COLLAB,
                        "item": WEIGHT_ITEM,
                        "user_text": WEIGHT_USER_TEXT,
                        "category": WEIGHT_CATEGORY,
                        "pop": WEIGHT_POP,
                    },
                    "include_self_posts": INCLUDE_SELF_POSTS_IN_FEED,
                    "map_hybrid_to_rank": MAP_HYBRID_TO_RANK
                }
            }), 200

        return jsonify(posts), 200

    except Exception as ex:
        import traceback
        ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        _append_rec_log([f"[{ts}][/ai/recommend][ERROR] {ex} {traceback.format_exc()}"])
        return jsonify({"error": "Internal Server Error"}), 500


def _inject_new_today(
    ids_all: List[int],
    user_id: int,
    positions: Optional[Tuple[int, ...]] = None,  # << เปลี่ยน: อนุญาต None เพื่อให้สุ่ม
    hours: int = 24,
    insert_cap: Optional[int] = None
) -> List[int]:
    """
    แทรก 'โพสต์ใหม่' พร้อม story log: รายชื่อ candidates + น้ำหนัก base/eng/recency + chosen
    - ถ้า positions=None => จะ 'สุ่ม' ตำแหน่งช่วง 21–30 (1-based) ตาม seed รายวันของ user
    - ถ้า positions ถูกส่งมา (tuple ของเลข 1-based) จะใช้ตามนั้น
    """
    try:
        e = _eng()

        content_df = _load_content_view(e)
        events_all = _load_events_all(e)

        ev_user = events_all[events_all.get("user_id") == int(user_id)] if "user_id" in events_all.columns else events_all.iloc[:0]
        interacted_set = set(pd.to_numeric(ev_user.get("post_id"), errors="coerce").dropna().astype(int).tolist()) if not ev_user.empty else set()

        seen_block = new_injected_seen_blocklist.get(int(user_id), set()).copy()

        try:
            self_posts = set(_get_authored_ids(e, int(user_id)))
        except Exception:
            self_posts = set()

        # เวลา TH สำหรับหน้าต่างโพสต์ใหม่
        try:
            from zoneinfo import ZoneInfo
            now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
        except Exception:
            now_th = datetime.utcnow() + timedelta(hours=7)
        cutoff = now_th - timedelta(hours=int(hours))

        # โหลด posts table เบื้องต้น
        try:
            sql = sqltext(f"""
                SELECT id, user_id, created_at, updated_at, status
                FROM {POSTS_TABLE}
            """)
            posts_df = pd.read_sql(sql, e)
        except Exception:
            posts_df = pd.DataFrame(columns=["id","user_id","created_at","updated_at","status"])

        if posts_df.empty:
            _append_rec_log([f"[{_fmt_th(_now_th())}][inject][WARN] posts table empty — skip injection"])
            return ids_all

        posts_df["id"] = pd.to_numeric(posts_df["id"], errors="coerce").astype("Int64")
        posts_df = posts_df.dropna(subset=["id"]).copy()
        posts_df["id"] = posts_df["id"].astype(int)

        if "status" in posts_df.columns:
            try:
                mask_active = (posts_df["status"].astype(str).str.lower() == "active") | (posts_df["status"].isna())
                posts_df = posts_df.loc[mask_active].copy()
            except Exception:
                pass

        def _pick_ts(row):
            cand = None
            for c in ("updated_at","created_at"):
                if c in row and pd.notna(row[c]):
                    t = pd.to_datetime(row[c], errors="coerce")
                    if not pd.isna(t):
                        cand = t; break
            return cand if cand is not None else pd.NaT

        posts_df["ts"] = posts_df.apply(_pick_ts, axis=1)

        def _to_th(ts: pd.Timestamp) -> Optional[datetime]:
            try:
                if pd.isna(ts): return None
                t = pd.Timestamp(ts)
                if t.tzinfo is None:
                    return t.tz_localize("Asia/Bangkok").to_pydatetime()
                return t.tz_convert("Asia/Bangkok").to_pydatetime()
            except Exception:
                return None

        posts_df["ts_th"] = posts_df["ts"].apply(_to_th)

        def _eligible_common(pid: int, uid_owner: Optional[int]) -> bool:
            if pid in interacted_set: return False
            if pid in seen_block: return False
            if not INCLUDE_SELF_POSTS_IN_FEED and uid_owner is not None:
                try:
                    if int(uid_owner) == int(user_id):
                        return False
                except Exception:
                    pass
            return True

        def _eligible_window(row) -> bool:
            pid = int(row["id"])
            tth = row["ts_th"]
            if tth is None or tth < cutoff:
                return False
            return _eligible_common(pid, row.get("user_id") if "user_id" in row else None)

        cand_df = posts_df.loc[posts_df.apply(_eligible_window, axis=1)].copy()

        fallback_used = False
        if cand_df.empty:
            tmp = posts_df.copy()
            tmp["ts_th_sort"] = tmp["ts_th"].apply(lambda x: x if isinstance(x, datetime) else datetime(1970,1,1))
            tmp = tmp.sort_values("ts_th_sort", ascending=False)
            fallback_ids = []
            for _, r in tmp.iterrows():
                pid = int(r["id"])
                if _eligible_common(pid, r.get("user_id") if "user_id" in r else None):
                    fallback_ids.append(pid)
                if len(fallback_ids) >= min(NEW_INSERT_MAX, 3):
                    break
            if not fallback_ids:
                _append_rec_log([f"[{_fmt_th(_now_th())}][inject] uid={user_id} candidates=0 (no-24h) fallback=0 positions={'random(21-30)' if positions is None else positions} window_h={hours}"])
                return ids_all
            cand_df = posts_df[posts_df["id"].isin(fallback_ids)].copy()
            fallback_used = True

        # คำนวณน้ำหนักเลือก candidates (base+eng+recency)
        try:
            base_scores = _compute_basescore_for(
                user_id=int(user_id),
                content_df=content_df,
                user_events=ev_user if not ev_user.empty else events_all.iloc[:0],
                precomputed_base_score=None,
                hybrid_override_C=None,
                hybrid_override_T=None,
            )
        except Exception:
            base_scores = {}

        try:
            eng_map = dict(zip(
                content_df["post_id"].astype(int).tolist(),
                pd.to_numeric(content_df[ENGAGE_COL], errors="coerce").fillna(0.0).astype(float).tolist()
            ))
        except Exception:
            eng_map = {}

        def _recency_norm(th: datetime) -> float:
            try:
                if fallback_used:
                    span = 7*24*3600.0
                    age  = (now_th - th).total_seconds() if isinstance(th, datetime) else span
                    val  = 1.0 - max(0.0, min(1.0, age / span))
                    return float(val)
                else:
                    span = (now_th - cutoff).total_seconds()
                    age  = (now_th - th).total_seconds()
                    val  = 1.0 - max(0.0, min(1.0, age / max(span, 1.0)))
                    return float(val)
            except Exception:
                return 0.0

        def _norm(series_vals: List[float]) -> Dict[int, float]:
            if not series_vals:
                return {}
            s = pd.Series(series_vals, dtype=float)
            mn, mx = float(s.min()), float(s.max())
            if mx - mn <= 1e-12:
                return {}
            return {i: (float(v) - mn) / (mx - mn) for i, v in enumerate(series_vals)}

        cand_ids = cand_df["id"].astype(int).tolist()
        raw_base, raw_eng, raw_rec = [], [], []
        tth_map = {}
        for pid in cand_ids:
            raw_base.append(float(base_scores.get(int(pid), 0.0)))
            raw_eng.append(float(eng_map.get(int(pid), 0.0)))
            tth = cand_df.loc[cand_df["id"] == pid, "ts_th"].values[0]
            tth_map[int(pid)] = tth if isinstance(tth, datetime) else None
            raw_rec.append(_recency_norm(tth) if isinstance(tth, datetime) else 0.0)

        nmap_base = _norm(raw_base)
        nmap_eng  = _norm(raw_eng)

        weights = []
        per_item_log = []
        for i, pid in enumerate(cand_ids):
            b = nmap_base.get(i, 0.0)
            g = nmap_eng.get(i, 0.0)
            r = raw_rec[i]
            w = 0.65*b + 0.25*g + 0.10*r
            if pid in set(ids_all):
                w += 0.02
            weights.append(max(0.0, float(w)))
            per_item_log.append({
                "pid": int(pid),
                "base_norm": float(b),
                "eng_norm": float(g),
                "rec_norm": float(r),
                "weight": float(weights[-1]),
                "ts_th": tth_map.get(int(pid)),
                "in_feed_before": bool(pid in set(ids_all))
            })

        # จำนวนที่จะฉีด
        cap = int(insert_cap) if insert_cap is not None else int(NEW_INSERT_MAX)
        cap = max(0, min(cap, max(1, len(cand_ids))))  # อย่างน้อย 1 เมื่อมี candidate

        rng = random.Random(_daily_seed(int(user_id)))

        def _weighted_sample_without_replacement(items: List[int], wts: List[float], k: int) -> List[int]:
            pool = list(items); ww = list(wts); out = []
            for _ in range(min(k, len(pool))):
                s = sum(ww)
                if s <= 0:
                    j = rng.randrange(0, len(pool))
                else:
                    r = rng.random()*s
                    acc = 0.0
                    j = 0
                    for j in range(len(pool)):
                        acc += ww[j]
                        if r <= acc:
                            break
                out.append(pool.pop(j)); ww.pop(j)
            return out

        chosen = cand_ids[:cap] if len(cand_ids) <= cap else _weighted_sample_without_replacement(cand_ids, weights, cap)

        # ===== เลือก "ตำแหน่ง inject" =====
        # ถ้าไม่ได้ส่ง positions มา → สุ่มในช่วง 21–30 (1-based), ไม่ซ้ำ, เรียงจากน้อยไปมาก
        if positions is None:
            lo_1based, hi_1based = 21, 30
            # clamp hi ตามความยาวลิสต์ (อนุโลมให้แทรกปลายลิสต์ได้โดยมี clamp ตอน insert อีกชั้น)
            hi_1based = max(lo_1based, min(hi_1based, max(lo_1based, len(ids_all))))
            k = min(cap, hi_1based - lo_1based + 1)
            pool_pos = list(range(lo_1based, hi_1based + 1))
            if k <= 0 or not pool_pos:
                positions_local = tuple()
            else:
                positions_local = tuple(sorted(rng.sample(pool_pos, k)))
        else:
            positions_local = positions

        # ประกอบฟีด: เอา chosen ออกก่อนแล้วค่อย insert กลับตามตำแหน่ง
        base = [p for p in ids_all if p not in set(chosen)]
        for i, pid in enumerate(chosen):
            if not positions_local:
                # ถ้าไม่มีตำแหน่ง (เช่นฟีดสั้นเกิน) -> แปะท้าย
                base.append(int(pid))
                continue
            pos_1based = positions_local[min(i, len(positions_local)-1)]
            pos = max(0, int(pos_1based) - 1)
            pos = min(pos, len(base))
            base.insert(pos, int(pid))

        # STORY LOG
        try:
            _log_story("inject", {
                "uid": int(user_id),
                "window_h": int(hours),
                "fallback": "latest" if fallback_used else "none",
                "positions": list(positions_local) if positions_local else [],
                "mode": "random(21-30)" if positions is None else "fixed",
                "candidates": per_item_log[:LOG_TOPN_PER_ITEM],
                "chosen": list(map(int, chosen)),
            })
        except Exception:
            pass

        _append_rec_log([
            f"[{_fmt_th(_now_th())}][inject] uid={user_id} "
            f"candidates={len(cand_ids)} chosen={chosen} "
            f"positions={(list(positions_local) if positions_local else [])} mode={'random' if positions is None else 'fixed'} window_h={hours} "
            f"{'fallback=latest' if fallback_used else 'fallback=none'}"
        ])
        return base

    except Exception as ex:
        _append_rec_log([f"[{_fmt_th(_now_th())}][inject][ERROR] {ex}"])
        return ids_all

def _now_th():
    try:
        return datetime.now(_TH_TZ)
    except Exception:
        # fallback: manual +7
        return datetime.utcnow() + timedelta(hours=7)

def _fmt_th(dt: datetime) -> str:
    # 2025-08-21 16:19:57 (ICT)
    return dt.strftime("%Y-%m-%d %H:%M:%S") + " ICT"

@verify_token
def ai_seen_handler():
    """
    POST /ai/seen
    body: { "seen_ids": [postId, ...] }
    - บันทึก impressions (TTL)
    - บันทึก blocklist สำหรับ new-post injection (ถ้าเห็นแล้ว ไม่ต้อง inject อีก)
    """
    try:
        body = _safe_get_body()
        uid = None
        if hasattr(request, "user_id") and request.user_id:
            uid = _as_int(request.user_id, 0)
        if not uid:
            uid = _as_int(body.get("user_id") or request.args.get("user_id"), 0)
        if not uid or uid <= 0:
            return jsonify({"error": "missing/invalid user_id"}), 400

        seen_ids = body.get("seen_ids") or []
        try:
            seen_ids = [int(x) for x in seen_ids if x is not None]
        except Exception:
            seen_ids = []

        if not seen_ids:
            return jsonify({"ok": True, "seen": 0}), 200

        # 1) TTL impressions
        _record_impressions(uid, seen_ids)

        # 2) Permanent blocklist สำหรับ new-post injection
        with _cache_lock:
            s = new_injected_seen_blocklist.get(int(uid))
            if s is None:
                s = set()
                new_injected_seen_blocklist[int(uid)] = s
            for pid in seen_ids:
                s.add(int(pid))

        ts = _fmt_th(_now_th())
        _append_rec_log([
            f"[{ts}][seen] uid={uid} seen_ids={seen_ids[:20]}{'...' if len(seen_ids)>20 else ''}"
        ])

        return jsonify({"ok": True, "seen": len(seen_ids)}), 200

    except Exception as ex:
        import traceback
        ts = _fmt_th(_now_th())
        _append_rec_log([f"[{ts}][/ai/seen][ERROR] {ex} {traceback.format_exc()}"])
        return jsonify({"error": "Internal Server Error"}), 500


def _split_seen_buckets(uid: int, ordered_ids: List[int], events_all: pd.DataFrame) -> Tuple[List[int], List[int], List[int]]:
    seen_recent_set = {int(h["post_id"]) for h in _get_impressions(uid)}

    pos_ev = events_all[
        (events_all["user_id"] == int(uid)) &
        (events_all["action_type"].isin(POS_ACTIONS))
    ]
    interacted_set = set(pos_ev["post_id"].astype(int).tolist())

    unseen, seen_no_pos, interacted = [], [], []
    for pid in ordered_ids:
        if pid in interacted_set:
            interacted.append(pid)
        elif pid in seen_recent_set:
            seen_no_pos.append(pid)
        else:
            unseen.append(pid)

    # LOG: split summary
    try:
        ts = _fmt_th(_now_th())
        _append_rec_log([f"[{ts}][split] uid={uid} total_input={len(ordered_ids)} unseen={len(unseen)} seen_no_pos={len(seen_no_pos)} interacted={len(interacted)}"])
    except Exception:
        pass

    return unseen, seen_no_pos, interacted

def _rebalance_by_category(
    order_ids: List[int],
    content_df: pd.DataFrame,
    base_scores: Optional[Dict[int, float]] = None,
    *,
    top_k: int = 20,
    cap_top: int = 3,
    cap_after: int = 3,
    window: int = 20,
    ratio_cap: float = 2.0,
    fairness_alpha: float = 0.22,     # บูสต์หมวดรองเล็กน้อย
    penalty_big: float = 0.60,        # โทษหนักเมื่อจะเกิน run-length cap
    penalty_ratio: float = 0.25,      # โทษเมื่อจะทำให้สัดส่วน > 2:1 ในหน้าต่าง
    prefer_top2_for_user: Optional[List[str]] = None,
    log_ctx: Optional[dict] = None,   # <<< NEW: injection context for story logs
) -> List[int]:
    """
    Rebalancer แบบ soft พร้อม story log ต่อการเลือกทุกตำแหน่ง:
      - จำกัด run-length ต่อหมวด (cap_top/cap_after)
      - บังคับ soft '2:1' ระหว่างหมวดท็อปสองในหน้าต่าง window
      - base_score - penalties + small boosts
    """
    if not order_ids:
        return []

    # สร้าง mapping post_id -> category
    pid_to_cat = {}
    for _, r in content_df.iterrows():
        try:
            pid = int(r["post_id"])
        except Exception:
            continue
        vals = r[CATEGORY_COLS].to_numpy(dtype=np.float32)
        if vals.size == 0:
            pid_to_cat[pid] = "Unknown"
        else:
            pid_to_cat[pid] = CATEGORY_COLS[int(np.argmax(vals))]

    # เตรียมคะแนนฐาน
    if base_scores is None:
        base_scores = {}
        n = len(order_ids)
        for rank, pid in enumerate(order_ids):
            base_scores[int(pid)] = float(n - rank) / max(1.0, n)

    # เตรียม candidate queue
    remaining = list(order_ids)

    # สำหรับ fairness window
    cat_window = deque(maxlen=max(1, int(window)))
    runlen_now = defaultdict(int)
    last_cat: Optional[str] = None

    out: List[int] = []
    steps_log: List[Dict[str, Any]] = []  # <<< เก็บเหตุผลรายตำแหน่ง

    def _cap_for_pos(pos: int) -> int:
        return int(cap_top) if pos < int(top_k) else int(cap_after)

    i = 0
    while remaining:
        cap_now = _cap_for_pos(i)
        lookahead = remaining[:60]

        best_j = None
        best_score = -1e9
        best_dbg = {
            "pen_runlen": False,
            "pen_ratio": False,
            "boost_secondary": False
        }

        # สถิติปัจจุบันใน window
        cnt_now = defaultdict(int)
        for c in cat_window:
            cnt_now[c] += 1

        for j, pid in enumerate(lookahead):
            c = pid_to_cat.get(int(pid), "Unknown")
            base = float(base_scores.get(int(pid), 0.0))
            score = base

            # 1) run-length penalty
            next_run = runlen_now[c] + 1 if c == last_cat else 1
            hit_run = (cap_now > 0 and next_run > cap_now)
            if hit_run:
                score -= penalty_big

            # 2) fairness ratio penalty (ลองอัปเดตหน้าต่างตามสมมติ)
            cnt2 = cnt_now.copy()
            cnt2[c] += 1
            cand_sorted = sorted(cnt2.items(), key=lambda x: x[1], reverse=True)
            hit_ratio = False
            if len(cand_sorted) >= 2:
                (c_top, n_top), (c_sec, n_sec) = cand_sorted[0], cand_sorted[1]
                if n_sec >= 1 and c == c_top and float(n_top) > float(ratio_cap) * float(n_sec):
                    overflow = float(n_top) - float(ratio_cap) * float(n_sec)
                    score -= penalty_ratio * (1.0 + 0.25 * overflow)
                    hit_ratio = True
            else:
                if (len(cand_sorted) == 1) and (cand_sorted[0][1] >= max(2, cap_now)):
                    score -= 0.10

            # 3) boost ให้หมวดรองของผู้ใช้
            boosted = False
            if prefer_top2_for_user and len(prefer_top2_for_user) >= 2:
                sec_name = prefer_top2_for_user[1]
                if c == sec_name:
                    score += float(fairness_alpha)
                    boosted = True

            if score > best_score:
                best_score = score
                best_j = j
                best_dbg = {
                    "pen_runlen": bool(hit_run),
                    "pen_ratio": bool(hit_ratio),
                    "boost_secondary": bool(boosted),
                    "base": float(base),
                    "score": float(score),
                    "cap_now": int(cap_now),
                }

        # เลือกผู้ชนะ
        if best_j is None:
            pid = remaining.pop(0)
        else:
            pid = remaining.pop(best_j)

        # อัปเดต out / run-length / window
        c = pid_to_cat.get(int(pid), "Unknown")
        if c == last_cat:
            runlen_now[c] += 1
        else:
            runlen_now = defaultdict(int)
            runlen_now[c] = 1
            last_cat = c
        cat_window.append(c)
        out.append(int(pid))

        # บันทึกเหตุผลตำแหน่งนี้
        try:
            steps_log.append({
                "pos": i + 1,
                "pid": int(pid),
                "cat": c,
                "cat_code": _cat_code(c),
                **best_dbg
            })
        except Exception:
            pass

        i += 1

    # push ลง log_ctx ถ้ามี
    if isinstance(log_ctx, dict):
        log_ctx.setdefault("rebalance", {})
        log_ctx["rebalance"]["steps"] = steps_log
        log_ctx["rebalance"]["params"] = {
            "top_k": int(top_k),
            "cap_top": int(cap_top),
            "cap_after": int(cap_after),
            "window": int(window),
            "ratio_cap": float(ratio_cap),
            "alpha": float(fairness_alpha),
        }

    return out

def _compute_basescore_for(
    user_id: int,
    content_df: pd.DataFrame,
    user_events: pd.DataFrame,
    precomputed_base_score: Optional[Dict[int, float]] = None,
    hybrid_override_C: Optional[Dict[int, float]] = None,
    hybrid_override_T: Optional[Dict[int, float]] = None,
) -> Dict[int, float]:
    now = datetime.utcnow()
    _vectorize_texts(content_df)
    user_prof      = _user_category_profile(user_id, user_events, content_df)
    follow_prof    = _follow_category_profile(_eng(), user_id, content_df)
    user_text_prof = _user_text_profile(user_id, user_events, content_df, _X)

    scores_E, scores_C, scores_F, scores_T, _scores_R = _build_scores(
        user_id, content_df, user_prof, follow_prof, user_text_prof, now
    )

    if MAP_HYBRID_TO_RANK:
        if hybrid_override_C:
            for k, v in hybrid_override_C.items():
                scores_C[int(k)] = float(v)
        if hybrid_override_T:
            for k, v in hybrid_override_T.items():
                scores_T[int(k)] = float(v)

    if precomputed_base_score:
        base_score = {int(pid): float(precomputed_base_score.get(int(pid), 0.0))
                      for pid in content_df["post_id"].astype(int)}
    else:
        def _final_score(E, C, F, T):
            return WEIGHT_E*E + WEIGHT_C*C + WEIGHT_F*F + WEIGHT_T*T
        base_score = {
            int(pid): _final_score(scores_E.get(pid, 0.0), scores_C.get(pid, 0.0),
                                   scores_F.get(pid, 0.0), scores_T.get(pid, 0.0))
            for pid in content_df["post_id"].astype(int)
        }
    return base_score

# ==================== SLIP & PROMPTPAY FUNCTIONS (from Slip.py) ====================

# === Order finder: ดึงคำสั่งซื้อด้วย ID (คืน dict หรือ None) ===
def find_order_by_id(order_id):
    try:
        order = Order.query.filter_by(id=order_id).first()
        if not order:
            return None
        return {
            'id': order.id,
            'user_id': order.user_id,
            'amount': order.amount,  # คง type เดิม (Numeric/Decimal) ไม่แก้พฤติกรรม
            'status': order.status,
            'promptpay_qr_payload': order.promptpay_qr_payload,
            'slip_image': order.slip_image,
            'renew_ads_id': order.renew_ads_id,
            'package_id': order.package_id,
            'show_at': order.show_at
        }
    except Exception as e:
        print(f"❌ [ERROR] find_order_by_id({order_id}): {e}")
        return None

# === Ad finder by order_id: ใช้ตรวจสถานะ/วันหมดอายุโฆษณาที่ผูกกับออเดอร์ ===
def find_ad_by_order_id(order_id):
    try:
        ad = Ad.query.filter_by(order_id=order_id).first()
        if not ad:
            return None
        return {
            'id': ad.id,
            'status': ad.status,
            'expiration_date': ad.expiration_date,
            'show_at': ad.show_at
        }
    except Exception as e:
        print(f"❌ [ERROR] find_ad_by_order_id({order_id}): {e}")
        return None

# === Ad finder by ad_id: ดึงรายละเอียดโฆษณาเต็มก้อน ===
def find_ad_by_id(ad_id):
    try:
        ad = Ad.query.filter_by(id=ad_id).first()
        if not ad:
            return None
        return {
            'id': ad.id,
            'user_id': ad.user_id,
            'order_id': ad.order_id,
            'title': ad.title,
            'content': ad.content,
            'link': ad.link,
            'image': ad.image,
            'status': ad.status,
            'expiration_date': ad.expiration_date,
            'created_at': ad.created_at,
            'updated_at': ad.updated_at,
            'show_at': ad.show_at
        }
    except Exception as e:
        print(f"❌ [ERROR] find_ad_by_id({ad_id}): {e}")
        return None

# === Package duration getter: ใช้คำนวณวันหมดอายุจากแพ็กเกจ ===
def get_ad_package_duration(package_id):
    try:
        pkg = AdPackage.query.filter_by(package_id=package_id).first()
        if not pkg:
            print(f"❌ [ERROR] AdPackage with ID {package_id} not found.")
            return None
        return pkg.duration_days
    except Exception as e:
        print(f"❌ [ERROR] get_ad_package_duration({package_id}): {e}")
        return None

# === Order status/slip updater: อัปเดตสถานะ + เก็บ path สลิป (ทรานแซกชันสั้นๆ) ===
def update_status_and_slip_info(order_id, new_status, slip_image_path, slip_transaction_id):
    try:
        order = Order.query.filter_by(id=order_id).first()
        if not order:
            print(f"❌ [ERROR] Order ID {order_id} not found for status update.")
            return False

        # ลดการเขียนซ้ำถ้าไม่มีอะไรเปลี่ยน (ไม่กระทบผลลัพธ์เดิม)
        changed = False
        if order.status != new_status:
            order.status = new_status
            changed = True
        if order.slip_image != slip_image_path:
            order.slip_image = slip_image_path
            changed = True

        if not changed:
            print(f"ℹ️ Order ID: {order_id} no changes applied.")
            return True

        order.updated_at = datetime.now()
        db.session.commit()
        print(f"✅ Order ID: {order_id} status updated to '{new_status}' with slip info.")
        return True
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error updating order status for ID {order_id}: {e}")
        return False

# === Ad status updater: เปลี่ยนสถานะโฆษณาแบบจงใจ ไม่ยุ่งวันหมดอายุ/วันเริ่ม ===
def update_ad_status(ad_id, new_status):
    try:
        ad = Ad.query.filter_by(id=ad_id).first()
        if not ad:
            print(f"❌ [ERROR] Ad ID {ad_id} not found for status update.")
            return False

        if ad.status == new_status:
            print(f"ℹ️ Ad ID: {ad_id} already in status '{new_status}'.")
            return True

        ad.status = new_status
        ad.updated_at = datetime.now()
        db.session.commit()
        print(f"✅ Ad ID: {ad_id} status updated to '{new_status}'.")
        return True
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error updating ad status for ID {ad_id}: {e}")
        return False

# === Ad renew updater: ต่ออายุ + เปลี่ยนสถานะ (คุมวันที่หมดอายุ) ===
def update_ad_for_renewal(ad_id, new_status, new_expiration_date):
    try:
        ad = Ad.query.filter_by(id=ad_id).first()
        if not ad:
            print(f"❌ [ERROR] Ad ID {ad_id} not found for renewal update.")
            return False

        changed = False
        if ad.status != new_status:
            ad.status = new_status
            changed = True
        if ad.expiration_date != new_expiration_date:
            ad.expiration_date = new_expiration_date
            changed = True

        if not changed:
            print(f"ℹ️ Ad ID: {ad_id} no changes applied (status/expiration unchanged).")
            return True

        ad.updated_at = datetime.now()
        db.session.commit()
        print(f"✅ Ad ID: {ad_id} status updated to '{new_status}' and expiration date extended to {new_expiration_date.strftime('%Y-%m-%d')}.")
        return True
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error updating ad for renewal ID {ad_id}: {e}")
        return False

# === Store PromptPay payload on Order: เก็บ QR payload ไว้อ้างอิงภายหลัง ===
def update_order_with_promptpay_payload_db(order_id, payload_to_store_in_db):
    try:
        order = Order.query.filter_by(id=order_id).first()
        if not order:
            print(f"❌ [ERROR] Order ID {order_id} not found for payload update.")
            return False

        if order.promptpay_qr_payload == payload_to_store_in_db:
            print(f"ℹ️ Order ID: {order_id} payload unchanged.")
            return True

        order.promptpay_qr_payload = payload_to_store_in_db
        order.updated_at = datetime.now()
        db.session.commit()
        print(f"✅ Order ID: {order_id} updated with PromptPay payload.")
        return True
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error updating order with PromptPay payload: {e}")
        return False

# === Create Ad by paid Order: สร้างโฆษณาใหม่ที่ผูกกับออเดอร์ที่ชำระแล้ว ===
def create_advertisement_db(order_data):
    try:
        now = datetime.now()
        default_title = f"Advertisement for Order {order_data['id']}"
        default_content = "This is a new advertisement pending admin approval after payment."
        ad_show_at = order_data.get('show_at', now)

        ad = Ad(
            user_id=order_data['user_id'],
            order_id=order_data['id'],
            title=default_title,
            content=default_content,
            link="",
            image="",
            status='paid',
            created_at=now,
            updated_at=now,
            show_at=ad_show_at
        )
        db.session.add(ad)
        db.session.commit()
        print(f"🚀 Advertisement ID: {ad.id} created for Order ID: {order_data['id']} with status 'paid'.")
        return ad.id
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error creating advertisement for Order ID {order_data.get('id')}: {e}")
        return None

# ===== ADD THIS AT THE VERY END OF THE FILE =====
def create_app() -> Flask:
    app = Flask(__name__)

    # Wire handlers
    app.add_url_rule("/ai/recommend", view_func=ai_recommend_handler, methods=["POST"])
    app.add_url_rule("/ai/seen", view_func=ai_seen_handler, methods=["POST"])

    @app.get("/healthz")
    def _healthz():
        # ใช้ CODE_VERSION_HASH ถ้ามีในไฟล์เพื่อเช็คเวอร์ชันโค้ด
        return jsonify({
            "ok": True,
            "service": "recsys",
            "db": "configured",
            "mode": os.getenv("APP_MODE", "dev"),
            "code_hash": globals().get("CODE_VERSION_HASH", "na"),
        })

    return app


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    debug = str(os.getenv("DEBUG", "false")).strip().lower() in ("1","true","yes","on")

    app = create_app()
    app.run(host=host, port=port, debug=debug)
