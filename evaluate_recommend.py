# ==================== RECOMMENDATION SYSTEM FUNCTIONS ====================
from transformers import AutoImageProcessor, SiglipForImageClassification
from apscheduler.schedulers.background import BackgroundScheduler
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sqlalchemy import Enum as SAEnum, create_engine
from sqlalchemy.sql import text
from flask_sqlalchemy import SQLAlchemy
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as sk_normalize
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from torchvision import models, transforms
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from selenium import webdriver
from surprise import SVD, Dataset, Reader
from qrcode.constants import ERROR_CORRECT_H
from textblob import TextBlob
from functools import wraps
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pythainlp.tokenize import word_tokenize
from PIL import Image, ImageOps, ImageFile
from datetime import datetime, date, timedelta, timezone

from sqlalchemy import create_engine, text
from scipy.sparse import csr_matrix, save_npz, load_npz
import os, json, pickle, hashlib, random
from typing import Optional, List, Dict, Tuple

import sys
import re
import io
import time
import json
import pytz
import uuid
import jwt
import torch
import torch.nn as nn
import pickle
import base64
import random
import locale
import secrets
import joblib
import traceback
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import qrcode

import os, time, math, json, pickle, random, hashlib, threading
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from sqlalchemy import create_engine
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz, save_npz

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
INCLUDE_SELF_POSTS_IN_FEED = True   # ถ้า False จะซ่อนโพสต์ที่ user เป็นเจ้าของจาก feed
USE_AUTHORED_AS_SIGNALS     = True
AUTHORED_CATEGORY_BONUS    = 0.7    # น้ำหนักเสริมสำหรับ profile category จากโพสต์ที่เขียนเอง
AUTHORED_TEXT_BONUS        = 0.7    # น้ำหนักเสริมสำหรับ text-profile จากโพสต์ที่เขียนเอง

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
    "item": 0.18,       # item-content / neighbor score weight
    "user_text": 0.12,  # user text-profile similarity weight
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
USE_HYBRID = False            # ถ้า True จะพยายามคำนวณ hybrid และใช้เป็น base_score ใน _rank
MAP_HYBRID_TO_RANK = True     # ถ้า True จะแมป category/text จาก HYBRID -> WEIGHT_C / WEIGHT_T

# -------------------- _rank-level WEIGHTS (engage/follow/recency) -----------
# _rank ใช้สัญญาณระดับโพสต์เพิ่มเติม (แยกจาก HYBRID)
WEIGHT_E = 0.30   # engagement (quality/viral)
WEIGHT_C = 0.40   # category match (จะถูก override เมื่อ MAP_HYBRID_TO_RANK=True)
WEIGHT_F = 0.12   # follow-influence category
WEIGHT_T = 0.12   # text relevance (จะถูก override เมื่อนำ HYBRID มาแมป)
WEIGHT_R = 0.06   # recency (ใช้เฉพาะ zone new / 21-30)

# ---------------- TF-IDF / ITEM-CONTENT / KNN params -----------------------
TFIDF_PARAMS = dict(
    analyzer="char_wb",
    ngram_range=(2, 5),
    max_features=60000,
    min_df=2,
    max_df=0.95,
)
KNN_NEIGHBORS = 10  # จำนวน neighbor ที่ใช้ใน KNN

# ---------------------- POPULARITY / SMOOTHING -----------------------------
POP_ALPHA = 5.0  # Bayesian smoothing สำหรับ PopularityPrior (ช่วยโพสต์ใหม่ไม่โดนลดจนน่าเกลียด)

# ---------------- Cache / impression TTL / directories ---------------------
OUT_DIR = "./recsys_eval_final"
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
MMR_LAMBDA = 0.80        # MMR lambda (ใกล้ 1 => เน้น relevance)
MMR_MAX_REF = 30         # จำนวน ref item เพื่อคำนวณ diversity penalty

NEW_WINDOWS_HOURS = [1, 3, 24]  # หน้าต่างเวลา (ชั่วโมง) สำหรับนิยาม "โพสต์ใหม่"
NEW_INSERT_MAX = 3               # จำนวนโพสต์ใหม่สูงสุดที่จะแทรกใน zone 21–30

CAT_MATCH_TOP20 = 0.60   # threshold ของ category-sim สำหรับ Top20
CAT_MATCH_AFTER = 0.50   # threshold หลัง Top20
ENG_PCTL_TOP20 = 40      # percentile ของ engagement สำหรับ Top20 (e.g., 40th)
ENG_PCTL_NEW = 25        # percentile สำหรับโพสต์ใหม่

# randomization temps (biased shuffle)
TEMP_UNSEEN = 0.15
TEMP_SEENNO = 0.12
TEMP_INTER = 0.10

# ---------------- global caches / lazy artifacts (internal state) ----------
# ตัวแปรพวกนี้เป็น state ของ process — เก็บไว้ใน module scope
recommendation_cache: Dict[int, Dict] = {}
impression_history_cache: Dict[int, List[Dict]] = {}
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

# --------------------------- SHORT GUIDANCE --------------------------------
# - ถ้าจะ "ยกหมวดหมู่" ให้ปรับ HYBRID_WEIGHTS['category'] (หรือ WEIGHT_CATEGORY ถาจะไม่ใช้ dict)
# - ถ้าต้องการ tune ให้เป็นเอกภาพ: ปรับ HYBRID_WEIGHTS แล้วตั้ง MAP_HYBRID_TO_RANK=True
# - ปรับ USE_HYBRID=True เฉพาะเมื่อเตรียม artifacts (tfidf/knn/svd) ไว้แล้วหรืออนุญาตให้สร้าง background
# ============================================================================ 



# ================================ UTILITIES =====================================
def _eng():
    return create_engine(DB_URI, pool_pre_ping=True, pool_recycle=1800)

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


def _log_recommendation(uid: int, start: int, page_size: int, return_all: bool, posts: List[dict]):
    try:
        ts = _fmt_th(_now_th())
        ids = [int(p["id"]) for p in posts]
        cats = []
        try:
            e = _eng(); content_df = _load_content_view(e)
            idx = content_df.set_index("post_id")
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
    now = datetime.utcnow()
    hist = impression_history_cache.get(user_id, [])
    hist = [h for h in hist if (now - h["ts"]).total_seconds() < IMPRESSION_HISTORY_TTL_SECONDS]
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
    """
    เวอร์ชันสุดท้าย (patched):
      - ใช้ _rank เป็นตัวจัดลำดับหลัก แต่สามารถเลือกที่จะใช้ hybrid precomputed base_score
        ถ้า USE_HYBRID=True จะพยายาม (best-effort) เตรียม artifacts แล้วคำนวณ scores
      - cache key ผูกกับ snapshot ของ content/events ด้วย _md5_of_df
      - นโยบาย seen filter คุมด้วย USE_TTL_SEEN (True=TTL impression, False=event-based)
      - เพิ่ม robustness กับ content_df ว่าง และป้องกัน exception ที่หนักใน request path
    """
    now = datetime.utcnow()

    # โหลดข้อมูลจำเป็น
    e = _eng()
    content_df = _load_content_view(e)
    events_all = _load_events_all(e)

    # ถ้าไม่มีโพสต์เลย ให้คืนว่าง
    if content_df is None or content_df.empty:
        return []

    # --- สร้าง cache key จากข้อมูลจริง (เปลี่ยนข้อมูลเมื่อไร key เปลี่ยน) ---
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

    cache_key = f"{uid}:{content_hash}:{events_hash}"

    # --- ตรวจ cache โดยผูกกับ key (ถ้า key ไม่ตรง แคชถือว่า invalid) ---
    with _cache_lock:
        cached = recommendation_cache.get(uid)
        if use_cache and cached and cached.get("key") == cache_key and \
           (now - cached.get("timestamp", now)).total_seconds() < CACHE_EXPIRY_TIME_SECONDS:
            return [int(x) for x in cached["ids"]]

    # events ของ user
    user_events = events_all[events_all["user_id"] == int(uid)] if "user_id" in events_all.columns else events_all.iloc[:0]

    # --- แยกกลุ่มสำหรับ _rank ตามนโยบายที่เลือก ---
    if USE_TTL_SEEN:
        all_ids = [int(x) for x in pd.to_numeric(content_df["post_id"], errors="coerce").dropna().astype(int).tolist()]
        unseen, seen_no, interacted = _split_seen_buckets(int(uid), all_ids, events_all)
    else:
        unseen, seen_no, interacted = _split_to_unseen_seenno_interacted(int(uid), content_df, events_all)

    # --- (ใหม่) ถ้าเปิด HYBRID -> เตรียม precomputed_base_score (best-effort) ---
    precomputed_scores = None
    if USE_HYBRID:
        # spawn background build (best-effort) so artifacts can be created outside request
        try:
            ensure_models_built(content_df, events_all, cache_key, cache_dir=CACHE_DIR, non_blocking=True)
        except Exception:
            pass

        # best-effort: try to obtain TF-IDF/X/postidx (may use global lazy-build)
        try:
            tfidf, X, postidx, _ = build_contentbased_models(content_df)
        except Exception:
            # fallback to globals if available
            tfidf, X, postidx = (_tfidf, _X, _postidx)

        # best-effort: precompute item_scores (if X available)
        try:
            if X is not None and getattr(X, "shape", (0,))[0] > 0:
                try:
                    knn = _build_knn(X)
                    item_scores = _precompute_item_content_scores(knn, content_df, X)
                except Exception:
                    # graceful fallback: zeros
                    item_scores = np.zeros(X.shape[0], dtype=np.float32)
            else:
                item_scores = np.zeros(len(content_df), dtype=np.float32)
        except Exception:
            item_scores = np.zeros(len(content_df), dtype=np.float32)

        # build train_pos -> user_text_profiles (best-effort)
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

        # collab model: best-effort (may be None)
        try:
            collab_model = build_collaborative_model(events_all, content_df["post_id"].astype(int).tolist())
        except Exception:
            collab_model = None

        # user category profile
        try:
            user_cat_prof = _user_category_profile(int(uid), user_events, content_df)
        except Exception:
            user_cat_prof = np.zeros(len(CATEGORY_COLS), dtype=np.float32)

        # compute hybrid dataframe (best-effort)
        try:
            df_hybrid = compute_hybridrecommendation_scores(
                int(uid), content_df, tfidf, X, postidx,
                user_text_profiles, collab_model, item_scores, user_cat_prof
            )
            # use 'final' (not final_norm) so absolute ordering preserved for ranking
            precomputed_scores = dict(zip(df_hybrid["post_id"].astype(int).tolist(),
                                         df_hybrid["final"].astype(float).tolist()))
        except Exception:
            precomputed_scores = None

    # --- จัดลำดับด้วย _rank (logic หลัก) ---
    try:
        ranked = _rank(int(uid), content_df, user_events, unseen, seen_no, interacted, precomputed_base_score=precomputed_scores)
    except Exception:
        # fallback safe: return simple popularity order if rank fails
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

    # --- อัปเดตแคชให้ผูกกับ key ปัจจุบัน ---
    with _cache_lock:
        recommendation_cache[uid] = {"ids": ranked, "timestamp": now, "key": cache_key}

    return [int(x) for x in ranked]


# ======================== DB FETCH (return full post objects) ====================
def fetch_posts_by_ids(ids: List[int], user_id: int) -> List[dict]:
    """ดึงโพสต์ตามลำดับ ids + user info + is_liked; เรียงตาม ids (ใช้ sqltext)"""
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

def _mmr_select(candidates: List[int], scores: Dict[int,float], simfunc, lam: float, k: int) -> List[int]:
    selected = []
    cand = list(dict.fromkeys(candidates))  # de-dupe preserve order
    while cand and len(selected) < k:
        best_id, best_val = None, -1e9
        for pid in cand:
            rel = scores.get(pid, 0.0)
            div = max(simfunc(pid, s) for s in selected[-MMR_MAX_REF:]) if selected else 0.0
            val = lam*rel - (1-lam)*div
            if val > best_val:
                best_val = val; best_id = pid
        if best_id is None: break
        selected.append(best_id)
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


def _rank(user_id: int, content_df: pd.DataFrame, user_events: pd.DataFrame,
          unseen: List[int], seen_no: List[int], interacted: List[int],
          precomputed_base_score: Optional[Dict[int, float]] = None) -> List[int]:
    """
    Ranking pipeline (patched):
      - รองรับ precomputed_base_score (จาก hybrid) ถ้ามี -> ใช้เป็น base_score
      - ถ้า pool (unseen + seen_no) ว่าง -> fallback เอา interacted มาใช้
      - เพิ่มความปลอดภัยในจุดที่อาจเกิด Index/empty errors
      - คืน ordered list ของ post_id (unique, preserved order)
    """
    now = datetime.now()

    # เตรียม TF-IDF / post-index (lazy)
    _vectorize_texts(content_df)

    # ==== โปรไฟล์ ====
    user_prof      = _user_category_profile(user_id, user_events, content_df)
    follow_prof    = _follow_category_profile(_eng(), user_id, content_df)
    user_text_prof = _user_text_profile(user_id, user_events, content_df, _X)

    # ==== สกอร์ฐาน ====
    scores_E, scores_C, scores_F, scores_T, scores_R = _build_scores(
        user_id, content_df, user_prof, follow_prof, user_text_prof, now
    )

    def _final_score(E, C, F, T, R, zone_is_new: bool) -> float:
        if zone_is_new:
            return WEIGHT_E*E + WEIGHT_C*C + WEIGHT_F*F + WEIGHT_T*T + WEIGHT_R*R
        return WEIGHT_E*E + WEIGHT_C*C + WEIGHT_F*F + WEIGHT_T*T

    # --- base_score: ถ้ามี precomputed_base_score ให้ใช้ (จาก hybrid) มิฉะนั้นคำนวณเดิม ---
    if precomputed_base_score:
        base_score = {int(pid): float(precomputed_base_score.get(int(pid), 0.0))
                      for pid in content_df["post_id"].astype(int)}
    else:
        base_score = {
            int(pid): _final_score(scores_E.get(int(pid), 0.0), scores_C.get(int(pid), 0.0), scores_F.get(int(pid), 0.0),
                                   scores_T.get(int(pid), 0.0), 0.0, zone_is_new=False)
            for pid in content_df["post_id"].astype(int)
        }

    # เตรียมข้อมูลช่วยตัดสิน
    ptiles = _category_percentiles_map(content_df)

    # set ของโพสต์ที่ตัวเองเป็นเจ้าของ (กัน/อนุญาตในฟีดตาม flag)
    try:
        self_post_ids = set(_get_authored_ids(_eng(), user_id))
    except Exception:
        self_post_ids = set()

    # ===================== Top 20 =====================
    # pool เป็น unseen + seen_no ตาม policy — แต่ถ้าว่าง fallback ให้ใช้ interacted
    pool = list(unseen) + list(seen_no)
    if not pool:
        # fallback: ถ้าผู้ใช้มี interacted แต่ไม่มี unseen/seen_no ให้ใช้ interacted เป็น pool
        pool = list(interacted)

    top20_cands = []
    for pid in pool:
        if (not INCLUDE_SELF_POSTS_IN_FEED) and (pid in self_post_ids):
            continue
        cat = category_by_pid(content_df, pid)
        ok_cat = scores_C.get(pid, 0.0) >= CAT_MATCH_TOP20
        # 안전하게อ่านค่า ENGAGE_COL
        try:
            eng_val = float(content_df.loc[content_df["post_id"] == pid, ENGAGE_COL].values[0])
        except Exception:
            eng_val = 0.0
        thr_map = ptiles.get(str(ENG_PCTL_TOP20), {})
        thr = thr_map.get(cat, thr_map.get("__global__", 0.0))
        ok_eng = eng_val >= thr
        if ok_cat and ok_eng:
            top20_cands.append(pid)

    if len(top20_cands) < 20:
        relax_pool = [pid for pid in pool if pid not in top20_cands]
        relax = sorted(
            relax_pool,
            key=lambda x: (scores_C.get(x, 0.0), base_score.get(x, 0.0)),
            reverse=True
        )
        for pid in relax:
            if (not INCLUDE_SELF_POSTS_IN_FEED) and (pid in self_post_ids):
                continue
            if len(top20_cands) >= 20:
                break
            top20_cands.append(pid)

    # MMR diversity ใน Top20
    def simfunc(a: int, b: int):
        ia, ib = _postidx.get(a, -1), _postidx.get(b, -1)
        if ia < 0 or ib < 0:
            return 0.0
        va, vb = _X[ia], _X[ib]
        num = float(va.multiply(vb).sum())
        den = (np.linalg.norm(va.data) * np.linalg.norm(vb.data)) if (hasattr(va, "data") and hasattr(vb, "data")) else 0.0
        return float(num/den) if den > 0 else 0.0

    top20_sorted = _mmr_select(
        candidates=sorted(set(top20_cands), key=lambda x: base_score.get(x, 0.0), reverse=True),
        scores=base_score, simfunc=simfunc, lam=MMR_LAMBDA, k=min(20, len(content_df))
    )

    top20_out, cat_seq = [], []
    for pid in top20_sorted:
        if (not INCLUDE_SELF_POSTS_IN_FEED) and (pid in self_post_ids):
            continue
        cat = category_by_pid(content_df, pid)
        if _runlen_violate(cat_seq, cat, RUNLEN_CAP_TOP20):
            continue
        top20_out.append(pid)
        cat_seq.append(cat)
        if len(top20_out) >= 20:
            break

    # ================= Zone 21–30 (ของใหม่) =================
    zone21_30 = []
    need_max = NEW_INSERT_MAX
    now_ts = now
    for h in NEW_WINDOWS_HOURS:
        new_cands = []
        for pid in unseen:
            if (not INCLUDE_SELF_POSTS_IN_FEED) and (pid in self_post_ids):
                continue
            row = content_df.loc[content_df["post_id"] == pid]
            if row.empty:
                continue
            ts = pd.NaT
            for c in ("created_ts", "created_at", "createdAt", "updated_at", "updatedAt", "timestamp"):
                if c in row.columns:
                    ts = pd.to_datetime(row[c].iloc[0], errors="coerce")
                    break
            if not _is_new(ts, now_ts, h):
                continue
            cat = category_by_pid(content_df, pid)
            cond_cat = scores_C.get(pid, 0.0) >= CAT_MATCH_AFTER
            thr_map_new = ptiles.get(str(ENG_PCTL_NEW), {})
            thr_new = thr_map_new.get(cat, thr_map_new.get("__global__", 0.0))
            try:
                eng_val = float(row[ENGAGE_COL].values[0])
            except Exception:
                eng_val = 0.0
            cond_eng = eng_val >= thr_new
            if cond_cat and cond_eng:
                sc = _final_score(
                    scores_E.get(pid, 0.0), scores_C.get(pid, 0.0), scores_F.get(pid, 0.0),
                    scores_T.get(pid, 0.0), scores_R.get(pid, 0.0), zone_is_new=True
                )
                new_cands.append((sc, pid))
        new_cands.sort(reverse=True)
        picked = [pid for _, pid in new_cands[:need_max]]
        if picked:
            zone21_30 = picked[:need_max]
            break

    # เติม 21–30 ด้วย best-of-rest (พร้อมเช็ก cap ต่อเนื่องจาก Top20)
    chosen20 = set(top20_out)
    chosen21 = set(zone21_30)
    remaining_pool = [pid for pid in unseen + seen_no if pid not in (chosen20 | chosen21)]
    if not INCLUDE_SELF_POSTS_IN_FEED:
        remaining_pool = [pid for pid in remaining_pool if pid not in self_post_ids]
    rest_sorted = sorted(remaining_pool, key=lambda x: base_score.get(x, 0.0), reverse=True)

    pos21_30 = []
    insert_positions = [22, 26, 29]

    # ต่อรันหมวดจาก Top20 มาเลย เพื่อให้ cap เป็น global ต่อเนื่อง
    cat_seq2 = cat_seq[:]  # cat_seq มาจาก Top20

    def _pick_from(lst: List[int]) -> Optional[int]:
        j = 0
        while j < len(lst):
            pid = lst[j]
            c = category_by_pid(content_df, pid)
            if not _runlen_violate(cat_seq2, c, RUNLEN_CAP_AFTER):
                lst.pop(j)
                cat_seq2.append(c)
                return pid
            j += 1
        return None

    for pos in range(21, 31):
        pid_choice = None
        # พยายามแทรก "ของใหม่" ตามตำแหน่งกำหนด โดยไม่ชน cap
        if pos in insert_positions and zone21_30:
            pid_choice = _pick_from(zone21_30)

        # ถ้ายังไม่มี (ชน cap หรือหมด) -> หยิบ best-of-rest แบบไม่ชน cap
        if pid_choice is None and rest_sorted:
            pid_choice = _pick_from(rest_sorted)

        if pid_choice is not None:
            pos21_30.append(pid_choice)
        # หากไม่มีตัวที่ไม่ชน cap เลย ให้เว้นไว้ก่อน (ช่วง tail จะเติมต่อด้วยกติกา cap เดิม)

    # ================= หลัง 30 =================
    chosen = set(top20_out + pos21_30)
    base_rest = [pid for pid in content_df["post_id"].astype(int) if pid not in chosen]
    if not INCLUDE_SELF_POSTS_IN_FEED:
        base_rest = [pid for pid in base_rest if pid not in self_post_ids]

    unseen_rest    = [pid for pid in unseen     if pid in base_rest]
    seenno_rest    = [pid for pid in seen_no    if pid in base_rest]
    interact_rest  = [pid for pid in interacted if pid in base_rest]

    # biased shuffle (daily seed)
    seed = _daily_seed(user_id)
    def _bs(ids, t, s): return _biased_shuffle(ids, base_score, t, s)
    unseen_rest   = _bs(unseen_rest,   TEMP_UNSEEN, seed+1)
    seenno_rest   = _bs(seenno_rest,   TEMP_SEENNO, seed+2)
    interact_rest = _bs(interact_rest, TEMP_INTER,  seed+3)

    U, S = len(unseen), len(seen_no)
    start_inter = _start_interacted_position(U, S)

    # ต่อเนื่องรันหมวด (รวม Top20 + 21–30) เพื่อรักษา cap แบบ global
    cat_seq_all = [category_by_pid(content_df, pid) for pid in (top20_out + pos21_30)]
    tail, pos_idx = [], len(top20_out) + len(pos21_30)

    # fill จนถึงจุดเริ่มแทรก interacted
    mix_pool = sorted(unseen_rest + seenno_rest, key=lambda x: base_score.get(x, 0.0), reverse=True)
    for pid in mix_pool:
        cat = category_by_pid(content_df, pid)
        if _runlen_violate(cat_seq_all, cat, RUNLEN_CAP_AFTER):
            continue
        tail.append(pid); cat_seq_all.append(cat)
        pos_idx += 1
        if pos_idx >= start_inter:
            break

    remain_ids  = [pid for pid in base_rest if pid not in set(tail)]
    remain_un   = [pid for pid in unseen_rest   if pid in remain_ids]
    remain_sn   = [pid for pid in seenno_rest   if pid in remain_ids]
    remain_it   = [pid for pid in interact_rest if pid in remain_ids]

    while remain_un or remain_sn or remain_it:
        block_items = []
        quota_inter = max(0, int(0.10 * 10))  # ~10% ต่อบล็อก 10
        for _ in range(10):
            best = None; best_score = -1.0
            pools = [
                ("un", remain_un),
                ("sn", remain_sn),
                ("it", remain_it if quota_inter > 0 else [])
            ]
            for name, pool in pools:
                if not pool:
                    continue
                cand = pool[0]
                s = base_score.get(cand, 0.0)
                if s > best_score:
                    best_score = s; best = (name, cand)
            if best is None:
                break

            name, cand = best
            cat = category_by_pid(content_df, cand)
            if _runlen_violate(cat_seq_all, cat, RUNLEN_CAP_AFTER):
                # ชน cap → ทิ้งหัวแถวนั้นแล้วพยายามตัวถัดไปในรอบหน้า
                pool = remain_un if name == "un" else remain_sn if name == "sn" else remain_it
                pool.pop(0)
                continue

            block_items.append(cand); cat_seq_all.append(cat)
            pool = remain_un if name == "un" else remain_sn if name == "sn" else remain_it
            pool.pop(0)
            if name == "it":
                quota_inter -= 1
            if len(block_items) >= 10:
                break

        if not block_items:
            # fallback: กวาดทั้งหมดอีกรอบแบบเคารพ cap
            leftovers = remain_un + remain_sn + remain_it
            moved = False
            for cand in leftovers:
                cat = category_by_pid(content_df, cand)
                if not _runlen_violate(cat_seq_all, cat, RUNLEN_CAP_AFTER):
                    block_items.append(cand)
                    cat_seq_all.append(cat)
                    if cand in remain_un: remain_un.remove(cand)
                    elif cand in remain_sn: remain_sn.remove(cand)
                    else:
                        if cand in remain_it: remain_it.remove(cand)
                    moved = True
            if not moved:
                # ถ้าจริงๆ ไม่มีอะไรวางได้โดยไม่ชน cap (rare) ให้ break ป้องกันลูปค้าง
                break

        tail.extend(block_items)

    final_list = top20_out + pos21_30 + tail
    ordered, seen_set = [], set()
    for pid in final_list:
        if pid not in seen_set:
            ordered.append(pid); seen_set.add(pid)
    return ordered

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
def ai_recommend_handler():
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

        # >>> เปลี่ยน default ให้ "คืนทั้งหมด" ถ้าไม่ได้ส่ง all มา <<<
        return_all = _as_bool(body.get("all") or request.args.get("all"), True)

        debug      = _as_bool(body.get("debug") or request.args.get("debug"), False)

        if refresh:
            with _cache_lock:
                recommendation_cache.pop(uid, None)

        ids_all = get_hybridrecommendation_order(uid, use_cache=(not refresh))
        total   = len(ids_all)

        # ---- เลือก candidate ids สำหรับ fetch (กันกรณีดรอปเพราะ post ไม่ active) ----
        def _candidate_ids_for_page(ids_all: List[int], start: int, page_size: int) -> List[int]:
            if return_all:
                seen=set(); out=[]
                for p in ids_all:
                    if p not in seen:
                        out.append(p); seen.add(p)
                return out
            # เก็บมากกว่าหน้าจริง 2x เผื่อดรอป
            seen=set(); out=[]
            i = max(0, start)
            target = page_size*2
            while i < len(ids_all) and len(out) < target:
                p = ids_all[i]
                if p not in seen:
                    out.append(p); seen.add(p)
                i += 1
            return out

        cand_ids = _candidate_ids_for_page(ids_all, start, page_size)
        posts = fetch_posts_by_ids(cand_ids, uid)

        # ถ้าไม่ใช่ all → ตัดให้เหลือ page_size ตามลำดับ ids_all[start:]
        if not return_all:
            mp = {int(p["id"]): p for p in posts}
            page_posts, i, seen_page = [], start, set()
            while len(page_posts) < page_size and i < len(ids_all):
                pid = int(ids_all[i])
                if pid not in seen_page and pid in mp:
                    page_posts.append(mp[pid]); seen_page.add(pid)
                i += 1
            posts = page_posts

        # -------- LOG ลงไฟล์ (ท้องถิ่น ICT) --------
        _log_recommendation(uid=uid, start=start, page_size=page_size,
                            return_all=return_all, posts=posts)

        # >>> สำคัญ: อย่า mark ว่า "เห็น" จากของที่ส่งออก <<<
        # (ลบ/งด _record_impressions(uid, [ids]) ตรงนี้)

        # [LOG-SUMMARY] ถ้าเปิดโหมดสรุป และเป็น refeed (refresh=true) -> พิมพ์สรุปจำนวนบรรทัด /ai/seen ที่ถูกซ่อน
        if SEEN_ACCESS_SUMMARY_ON_RECOMMEND and refresh:
            cnt = _seen_pop_count()
            if cnt > 0:
                # แสดงเวลาไทยให้อ่านง่าย
                try:
                    from zoneinfo import ZoneInfo
                    now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
                    ts = now_th.strftime("%Y-%m-%d %H:%M:%S ICT")
                except Exception:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{ts}] suppressed /ai/seen access logs: {cnt} lines since last refresh")

        # -------- RESPONSE --------
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
                    }
                }
            }), 200
        return jsonify(posts), 200

    except Exception as ex:
        import traceback
        ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        _append_rec_log([f"[{ts}][/ai/recommend][ERROR] {ex} {traceback.format_exc()}"])
        return jsonify({"error": "Internal Server Error"}), 500

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

        _record_impressions(uid, seen_ids)

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
    """
    แบ่งโพสต์เป็น 3 กลุ่มตาม priority:
      - unseen: ยังไม่เคยถูก 'ส่งให้ client' ในช่วง TTL (ดูจาก impression_history_cache)
      - seen_no_positive: เคยส่งให้ client แล้ว แต่ 'ไม่มี' POS_ACTIONS
      - interacted: มี POS_ACTIONS (like/comment/bookmark/share) กับ user นี้
    """
    # impressions ล่าสุดใน TTL
    seen_recent_set = {int(h["post_id"]) for h in _get_impressions(uid)}

    # โพสต์ที่มีปฏิสัมพันธ์เชิงบวก
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

    return unseen, seen_no_pos, interacted

# ======================= OFFLINE EVAL (Presentation Minimal, TH time) =======================
# รันไฟล์นี้ตรงๆ: python Recommend_Evaluation.py
# โหมดประเมิน: Business-fair (ตัด seen ออกก่อนคิดเมตริก), target จาก neighbors-proxy
# บันทึกไฟล์เฉพาะที่จำเป็น: summary_metrics.csv + PNG(precision/recall/ndcg) + run_info.json
# โฟลเดอร์ผลใช้เวลาไทย Asia/Bangkok: eval_YYYYMMDDTHHMMSS+0700

import os, json, math, sys, time, traceback
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Time helpers (Thai time) ----------
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

def _now_local():
    tz = ZoneInfo("Asia/Bangkok") if ZoneInfo else None
    return datetime.now(tz) if tz else datetime.now()

def _ts_for_dir_local():
    dt = _now_local()
    # รูปแบบที่ path-safe บน Windows: ไม่มีเครื่องหมาย :
    # ตัวอย่าง: 20250823T220812+0700
    return dt.strftime("%Y%m%dT%H%M%S%z") if dt.tzinfo else dt.strftime("%Y%m%dT%H%M%S")

def _now_str():
    dt = _now_local()
    return dt.strftime("%Y-%m-%d %H:%M:%S%z") if dt.tzinfo else dt.strftime("%Y-%m-%d %H:%M:%S")

# -------------------------- Small utils --------------------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True); return p

def _now():
    return _now_str()

def _log(step, msg, **kv):
    kvs = " ".join([f"{k}={v}" for k, v in kv.items()])
    print(f"[{_now()}][{step}] {msg} {kvs}".strip(), flush=True)

def _progress_bar(i, n, width=40):
    p = 0 if n <= 0 else int((i / n) * width)
    bar = "[" + "#" * p + "-" * (width - p) + "]"
    pct = 0 if n <= 0 else int((i / n) * 100)
    print(f"\r{bar} {pct}%", end="", flush=True)
    if i == n:
        print("", flush=True)

def _ap_at_k(rels, k):
    hits, sum_prec = 0, 0.0
    upto = min(k, len(rels))
    denom = 0
    for i in range(upto):
        if rels[i] > 0:
            hits += 1
            sum_prec += hits / float(i + 1)
            denom += 1
    return (sum_prec / max(denom, 1)) if upto > 0 else 0.0

def _ndcg_at_k(rels, k):
    def _dcg(arr):
        s = 0.0
        for i, r in enumerate(arr, start=1):
            s += (2 ** r - 1) / math.log2(i + 1)
        return s
    rels_k = rels[:k]
    dcg = _dcg(rels_k)
    ideal = sorted(rels, reverse=True)[:k]
    idcg = _dcg(ideal)
    return (dcg / idcg) if idcg > 1e-12 else 0.0

def _plot_bar_at_k(summary_rows: list, ks: list, metric: str, out_base: str, title=None, ylabel=None):
    xs = [str(k) for k in ks]
    ys = [float(next(r[metric] for r in summary_rows if r["k"] == k)) for k in ks]
    plt.figure(figsize=(7, 4), dpi=160)
    plt.bar(xs, ys)
    plt.title(title or f"{metric.upper()}@K")
    plt.xlabel("K")
    plt.ylabel(ylabel or metric)
    plt.tight_layout()
    fp = os.path.join(out_base, f"{metric}_at_k.png")
    try:
        plt.savefig(fp)
    finally:
        plt.close()

# -------------------------- Build sets --------------------------
def _build_positive_sets(events_all: pd.DataFrame) -> dict:
    """
    positive = action ∈ POS_ACTIONS หรือ view >= VIEW_POS_MIN
    ถ้ามี NEG_ACTIONS กับโพสต์นั้น จะไม่นับเป็นบวก
    """
    if events_all.empty:
        return {}
    t = events_all.groupby(["user_id", "post_id", "action_type"]).size().reset_index(name="cnt")
    pvt = t.pivot_table(
        index=["user_id", "post_id"], columns="action_type",
        values="cnt", fill_value=0, aggfunc="sum"
    ).reset_index()
    pvt.columns = [str(c).lower() for c in pvt.columns]

    pos = np.zeros(len(pvt), dtype=bool)
    for a in POS_ACTIONS:
        if a in pvt.columns:
            pos |= (pvt[a].to_numpy(dtype=float) > 0)
    if "view" in pvt.columns:
        pos |= (pvt["view"].to_numpy(dtype=float) >= float(VIEW_POS_MIN))
    if NEG_ACTIONS:
        neg = np.zeros(len(pvt), dtype=bool)
        for a in NEG_ACTIONS:
            if a in pvt.columns:
                neg |= (pvt[a].to_numpy(dtype=float) > 0)
        pos = np.where(neg, False, pos)

    positives = pvt[pos][["user_id", "post_id"]].copy()
    positives["user_id"] = pd.to_numeric(positives["user_id"], errors="coerce").astype(int)
    positives["post_id"] = pd.to_numeric(positives["post_id"], errors="coerce").astype(int)
    mp = {}
    for uid, grp in positives.groupby("user_id"):
        mp[int(uid)] = set(int(x) for x in grp["post_id"].tolist())
    return mp

def _build_seen_sets(events_all: pd.DataFrame) -> dict:
    """seen = มี interaction ใดๆ หรือ view >= 1"""
    if events_all.empty:
        return {}
    t = events_all.groupby(["user_id","post_id"]).size().reset_index(name="cnt_any")
    v = events_all[events_all["action_type"].str.lower()=="view"] if "action_type" in events_all.columns else events_all[:0]
    v = v.groupby(["user_id","post_id"]).size().reset_index(name="cnt_view") if not v.empty else pd.DataFrame(columns=["user_id","post_id","cnt_view"])
    m = pd.merge(t, v, on=["user_id","post_id"], how="left").fillna(0)
    m["user_id"]=pd.to_numeric(m["user_id"], errors="coerce").astype(int)
    m["post_id"]=pd.to_numeric(m["post_id"], errors="coerce").astype(int)
    seen={}
    for uid,g in m.groupby("user_id"):
        seen[int(uid)]=set(int(x) for x in g["post_id"].tolist())
    return seen

def _neighbor_proxy_targets(user_pos: dict,
                            jaccard_min: float = 0.1,
                            min_overlap: int = 1,
                            top_m_neighbors: int = 50) -> dict:
    """
    proxy_future[u] = ยูเนียนโพสต์บวกของเพื่อนบ้านที่คล้าย u - positives ของ u
    ไม่ใช้เวลาเลย และไม่ชน seen filter
    """
    users = list(user_pos.keys())
    proxy = {u: set() for u in users}
    for u in users:
        Su = user_pos.get(u, set())
        neigh_scores = []
        for v in users:
            if v == u:
                continue
            Sv = user_pos.get(v, set())
            inter = len(Su & Sv)
            if inter < min_overlap:
                continue
            union = len(Su | Sv)
            if union == 0:
                continue
            j = inter / float(union)
            if j >= jaccard_min:
                neigh_scores.append((j, v))
        neigh_scores.sort(reverse=True)
        neighs = [v for _, v in neigh_scores[:top_m_neighbors]]
        cand = set()
        for v in neighs:
            cand |= user_pos.get(v, set())
        proxy[u] = cand - Su
    return proxy

# -------------------------- Core evaluation (Business-fair only) --------------------------
def evaluate_recommender_presentation(
    ks=(5,10,20),
    jaccard_min=0.1,
    min_overlap=1,
    top_m_neighbors=50,
    seed=42,
    out_root=None,
    save_per_user: bool = bool(int(os.getenv("EVAL_SAVE_PER_USER","0")))
):
    """
    โหมดพรีเซนต์: ตัด seen ออกก่อนคิดเมตริก (สิ่งที่ UI จะโชว์จริง)
    บันทึก: summary_metrics.csv + PNG (precision/recall/ndcg) + run_info.json
    """
    start_ts = _now_local()

    # เตรียมโฟลเดอร์
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_root = out_root or os.path.join(script_dir, "RecEva")

    _log("0/8", "Start evaluation (presentation fair)", out_dir=out_root,
         jaccard_min=jaccard_min, min_overlap=min_overlap, top_m=top_m_neighbors)

    # 1) content
    _log("1/8", "Loading content view...")
    e = _eng()
    content_df = _load_content_view(e)
    all_pids = set(pd.to_numeric(content_df["post_id"], errors="coerce").dropna().astype(int).tolist())
    _log("1/8", "Loaded content view", rows=len(content_df))

    # 2) events
    _log("2/8", "Loading events...")
    events_all = _load_events_all(e)
    _log("2/8", "Loaded events", rows=len(events_all))

    # 3) sets
    _log("3/8", "Building positives & seen sets...")
    user_pos = _build_positive_sets(events_all)
    user_seen = _build_seen_sets(events_all)
    users_all = sorted([u for u, s in user_pos.items() if s])

    # 4) proxy targets
    _log("4/8", "Building neighbor proxy targets...")
    proxy = _neighbor_proxy_targets(user_pos, jaccard_min, min_overlap, top_m_neighbors)
    users = [u for u in users_all if len(proxy.get(u, set())) > 0]
    _log("4/8", "Users with targets", count=len(users))

    if not users:
        ts = _ts_for_dir_local()
        out_base = _ensure_dir(os.path.join(out_root, f"eval_{ts}"))
        pd.DataFrame([], columns=["k","users","precision","recall","f1","map","ndcg"])\
          .to_csv(os.path.join(out_base,"summary_metrics.csv"), index=False, encoding="utf-8")
        with open(os.path.join(out_base,"run_info.json"),"w",encoding="utf-8") as f:
            json.dump({"reason":"no users with proxy targets", "timezone":"Asia/Bangkok"}, f, ensure_ascii=False, indent=2)
        _log("DONE","No users with proxy targets.", path=out_base)
        return {"out_dir": out_base}

    # 5) eval loop
    _log("5/8", "Evaluating users (fair)...")
    rng = np.random.default_rng(seed)
    agg = {k: {"precision": [], "recall": [], "f1": [], "map": [], "ndcg": []} for k in ks}
    per_user_rows = []
    N = len(users)
    for idx, uid in enumerate(users, start=1):
        targets = proxy.get(uid, set())
        if not targets:
            continue

        try:
            rec_ids = get_hybridrecommendation_order(int(uid), use_cache=True)
            rec_ids = [int(x) for x in rec_ids]
        except Exception as ex:
            _log("WARN", f"Recommend failed for user {uid}, skip.", err=str(ex))
            continue

        # fair: กรอง seen ออก
        seen = user_seen.get(uid, set())
        rec_ids = [pid for pid in rec_ids if pid not in seen]

        for k in ks:
            rec_k = rec_ids[:k]
            rels = [1 if pid in targets else 0 for pid in rec_k]
            tp = int(sum(rels))

            prec = tp / float(max(1, k))
            rec  = tp / float(max(1, len(targets)))
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 1e-12 else 0.0
            mapk = _ap_at_k(rels, k)
            ndcg = _ndcg_at_k(rels, k)

            agg[k]["precision"].append(prec)
            agg[k]["recall"].append(rec)
            agg[k]["f1"].append(f1)
            agg[k]["map"].append(mapk)
            agg[k]["ndcg"].append(ndcg)

            if save_per_user:
                per_user_rows.append({
                    "user_id": int(uid), "k": int(k), "targets": int(len(targets)),
                    "precision": float(prec), "recall": float(rec), "f1": float(f1),
                    "map": float(mapk), "ndcg": float(ndcg)
                })

        _progress_bar(idx, N)

    # 6) สรุป
    ts = _ts_for_dir_local()
    out_base = _ensure_dir(os.path.join(out_root, f"eval_{ts}"))

    summary_rows = []
    for k in ks:
        n = len(agg[k]["precision"])
        summary_rows.append({
            "k": int(k),
            "users": int(n),
            "precision": float(np.mean(agg[k]["precision"])) if n else 0.0,
            "recall":    float(np.mean(agg[k]["recall"]))    if n else 0.0,
            "f1":        float(np.mean(agg[k]["f1"]))        if n else 0.0,
            "map":       float(np.mean(agg[k]["map"]))       if n else 0.0,
            "ndcg":      float(np.mean(agg[k]["ndcg"]))      if n else 0.0,
        })

    # 7) บันทึกไฟล์ (minimal set)
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_base, "summary_metrics.csv"), index=False, encoding="utf-8")
    if save_per_user and per_user_rows:
        pd.DataFrame(per_user_rows).to_csv(os.path.join(out_base, "per_user_metrics.csv"), index=False, encoding="utf-8")

    for m, ttl in [("precision","PRECISION@K"), ("recall","RECALL@K"), ("ndcg","NDCG@K")]:
        _plot_bar_at_k(summary_rows, list(ks), m, out_base, title=ttl, ylabel=m)

    # 8) run_info.json
    finished_at = _now_local()
    run_info = {
        "mode": "presentation_fair",
        "started_at_local": start_ts.isoformat(),
        "finished_at_local": finished_at.isoformat(),
        "timezone": "Asia/Bangkok",
        "params": {
            "ks": list(ks),
            "jaccard_min": jaccard_min,
            "min_overlap": min_overlap,
            "top_m_neighbors": top_m_neighbors,
            "save_per_user": bool(save_per_user),
        },
        "paths": {"out_dir": out_base},
        "summary": summary_rows
    }
    with open(os.path.join(out_base, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    _log("8/8", "Saved results", out_dir=out_base)
    print(json.dumps({"summary": summary_rows, "out_dir": out_base}, ensure_ascii=False, indent=2))
    return {"out_dir": out_base}

# -------------------------- ENTRY: run by default --------------------------
def _run_eval_presentation_main():
    try:
        evaluate_recommender_presentation(
            ks=tuple(int(x) for x in os.getenv("EVAL_KS","5,10,20").split(",")),
            jaccard_min=float(os.getenv("EVAL_JACCARD_MIN","0.1")),
            min_overlap=int(os.getenv("EVAL_MIN_OVERLAP","1")),
            top_m_neighbors=int(os.getenv("EVAL_TOP_M_NEIGHBORS","50")),
            seed=int(os.getenv("EVAL_SEED","42")),
            out_root=None,
            save_per_user=bool(int(os.getenv("EVAL_SAVE_PER_USER","0")))
        )
    except Exception as ex:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fail_dir = _ensure_dir(os.path.join(script_dir, "RecEva", "failed"))
        with open(os.path.join(fail_dir, f"error_{int(time.time())}.log"), "w", encoding="utf-8") as f:
            f.write(str(ex) + "\n\n" + traceback.format_exc())
        _log("ERROR", "Evaluation crashed", err=str(ex), log_dir=fail_dir)
        raise

if __name__ == "__main__":
    _run_eval_presentation_main()