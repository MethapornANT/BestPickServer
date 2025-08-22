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

app = Flask(__name__)

# กันรูปยักษ์/ไฟล์ขาดไม่ให้ทำโปรเซสเด้ง (ไม่กระทบผลลัพธ์เดิม)
Image.MAX_IMAGE_PIXELS = 25_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True

# === Config: upload path & model path ===
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = './NSFW_Model/results/20250817-022748/efficientnetb0/models/efficientnetb0_best.pth'

# === Threshold (เหมือนเดิม) ===
HENTAI_THRESHOLD = 37.0
PORN_THRESHOLD   = 41.0

CALIBRATION_MIX_ALPHA = 0.4

# === Labels: mapping id <-> label ===
LABELS = ['normal', 'hentai', 'porn', 'sexy', 'anime']
label2idx = {label: idx for idx, label in enumerate(LABELS)}
idx2label = {idx: label for label, idx in label2idx.items()}

# === Model: Custom EfficientNetB0 (โครงเดียวกับตอนเทรน) ===
class CustomEfficientNetB0(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomEfficientNetB0, self).__init__()
        # ใช้ weights=None/ pretrained=False ให้คงพฤติกรรมเดิม
        try:
            self.efficientnet = models.efficientnet_b0(weights=None)
        except TypeError:
            self.efficientnet = models.efficientnet_b0(pretrained=False)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)

# === Load model: พยายามโหลดอย่างอ่อนโยน ไม่ทำให้โปรเซสตาย (เหมือนเดิม) ===
model = CustomEfficientNetB0(num_classes=len(LABELS))
MODEL_READY = True
try:
    state_dict_from_file = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    try:
        # ถ้าไฟล์ .pth เซฟมาจาก backbone โดยตรง
        model.efficientnet.load_state_dict(state_dict_from_file)
    except Exception:
        # เผื่อ key ต่างเล็กน้อย แต่ยังให้วิ่งต่อได้
        model.load_state_dict(state_dict_from_file, strict=False)
    print("โหลดโมเดลสำเร็จ")
except Exception as e:
    print(f"โหลดโมเดลไม่สำเร็จ แต่จะไม่ปิดเซิร์ฟเวอร์: {e}")
    MODEL_READY = False

model.eval()

# === (ใหม่) Calibrator แบบ optional: โหลดได้ก็ใช้, โหลดไม่ได้ก็ข้าม ===
CALIBRATION_DIRS = [
    './NSFW_Evaluation/calibration_models_oof',
    './NSFW_Evaluation/calibration_models',
]
CALIBRATION_ENABLE = True  # จะเปิด/ปิดได้ด้วยบรรทัดเดียว

_isotonic_h = None
_isotonic_p = None
if CALIBRATION_ENABLE:
    def _try_load_iso(path):
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            # quick sanity
            _ = float(obj.transform([0.1])[0])
            return obj
        except Exception:
            return None

    for _d in CALIBRATION_DIRS:
        if _isotonic_h is None:
            ph = os.path.join(_d, 'final_isotonic_hentai.pkl')
            if os.path.isfile(ph):
                _isotonic_h = _try_load_iso(ph)
        if _isotonic_p is None:
            pp = os.path.join(_d, 'final_isotonic_porn.pkl')
            if os.path.isfile(pp):
                _isotonic_p = _try_load_iso(pp)
        if _isotonic_h is not None or _isotonic_p is not None:
            break
# หมายเหตุ: ไม่พิมพ์ log เพิ่ม เพื่อลดความรก

def _apply_iso(ir, v):
    # ปลอดภัย: ถ้าไม่มี/พัง -> คืนค่าเดิม
    try:
        if ir is None:
            return float(v)
        out = float(ir.transform([float(v)])[0])
        # clamp เผื่อ calibrator ให้ค่านอกช่วง
        return float(min(max(out, 0.0), 1.0))
    except Exception:
        return float(v)

# === Preprocess: ให้ตรงกับตอนเทรน + แก้ EXIF orientation ===
processor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# === Inference API (file path): ตรวจ NSFW ของภาพจากพาธไฟล์ ===
def nude_predict_image(image_path):
    """
    พฤติกรรมเดิมทุกอย่าง (signature/return/log เดิม)
    เพิ่ม calibration แบบ conservative: ใช้ max(raw, calibrated) ในการตัดสิน -> ไม่มีวันทำให้ปล่อยหลุดเพิ่ม
    """
    try:
        # เปิดภาพแบบแก้เอียงจาก EXIF + บังคับ RGB
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        inputs = processor(image).unsqueeze(0)

        # ส่งไปยัง device เดียวกับโมเดล (เดิมของคุณอยู่บน CPU ก็ปล่อยให้เป็น CPU)
        device = next(model.parameters()).device
        inputs = inputs.to(device)

        with torch.inference_mode():
            if MODEL_READY:
                outputs = model(inputs)
                logits = outputs
                probs = torch.softmax(logits, dim=1).squeeze().tolist()
            else:
                # กรณีโมเดลโหลดไม่ได้ ให้คืนศูนย์ทุกคลาสเพื่อไม่ทำให้ระบบพัง
                probs = [0.0] * len(LABELS)

        # กัน index พัง (ปกติไม่ควรเกิดถ้า LABELS = 5 คลาส)
        if label2idx['hentai'] >= len(probs) or label2idx['porn'] >= len(probs):
            raise IndexError("ไม่พบ Index ของ 'hentai' หรือ 'porn' ในผลลัพธ์การคาดการณ์")

        # raw probs (0..1)
        h_raw = float(probs[label2idx['hentai']])
        p_raw = float(probs[label2idx['porn']])

        # (ใหม่) calibrated probs (0..1) — ถ้าไม่มี calibrator จะได้ค่าเดิม
        h_cal = _apply_iso(_isotonic_h, h_raw)
        p_cal = _apply_iso(_isotonic_p, p_raw)

        h_final_pct = (CALIBRATION_MIX_ALPHA * h_cal + (1 - CALIBRATION_MIX_ALPHA) * h_raw) * 100.0
        p_final_pct = (CALIBRATION_MIX_ALPHA * p_cal + (1 - CALIBRATION_MIX_ALPHA) * p_raw) * 100.0

        # ตัดสินผลด้วยค่า final (จะ “เข้มขึ้น” หรือ “เท่าเดิม” เท่านั้น)
        is_nsfw = (h_final_pct >= HENTAI_THRESHOLD) or (p_final_pct >= PORN_THRESHOLD)

        # ---- LOG แบบเดิมเป๊ะ (โชว์ค่าดิบเพื่อไม่ทำ dev pipeline เปลี่ยน) ----
        # *ถ้าอยากดูค่า cal ให้เปิดคอมเมนต์สองบรรทัดด้านล่างแทน*
        print(f"NSFW Detection for {image_path}:")
        print(f"   Hentai: {h_raw*100.0:.2f}%")
        print(f"   Pornography: {p_raw*100.0:.2f}%")
        print(f"   Hentai(cal-used): {h_final_pct:.2f}% | Porn(cal-used): {p_final_pct:.2f}%")
        print(f"   Is NSFW: {is_nsfw} (thresholds: hentai>={HENTAI_THRESHOLD} | porn>={PORN_THRESHOLD})")

        # รวมคะแนนทุกคลาสเป็นเปอร์เซ็นต์ (คงรูปแบบเดิม)
        result_dict = {}
        for i in range(len(probs)):
            if i in idx2label:
                result_dict[idx2label[i]] = round(float(probs[i])*100.0, 2)
            else:
                result_dict[f"Class_{i}"] = round(float(probs[i])*100.0, 2)

        # ไม่เปลี่ยน payload เพื่อกันระบบพัง; calibration ใช้ภายในการตัดสินเท่านั้น
        if not MODEL_READY:
            result_dict["__degraded__"] = 1.0

        return bool(is_nsfw), result_dict

    except Exception as e:
        print(f"Error in NSFW detection for {image_path}: {e}")
        return False, {"error": f"ไม่สามารถตรวจสอบภาพได้: {e}"}


# ========== App/DB Config (ไม่เปลี่ยน URI เดิม) ==========
# กำหนดค่า URI สำหรับการเชื่อมต่อฐานข้อมูล
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:1234@localhost/bestpick'

# ลด overhead ของ SQLAlchemy และกันคอนเนคชันเน่า
app.config.setdefault('SQLALCHEMY_TRACK_MODIFICATIONS', False)
app.config.setdefault('SQLALCHEMY_ENGINE_OPTIONS', {
    'pool_pre_ping': True,     # ping ก่อนใช้กันคอนเนคชันตาย
    'pool_recycle': 1800,      # รีไซเคิลทุก 30 นาที กัน MySQL ตัดคอนเนคชัน
    'pool_size': 10,           # ปรับตามทรัพยากร
    'max_overflow': 20,
})

# เริ่มต้นใช้งาน SQLAlchemy
db = SQLAlchemy(app)

# ========== SQLAlchemy Models: Order / Ad / AdPackage ==========
# หมายเหตุ: ไม่เปลี่ยนชื่อตาราง/คอลัมน์/ชนิดข้อมูล เพื่อตรงกับของเดิม

# โมเดล Order (คำสั่งซื้อ)
class Order(db.Model):
    __tablename__ = 'orders'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)

    renew_ads_id = db.Column(db.Integer, nullable=True)
    package_id = db.Column(db.Integer, nullable=True)

    amount = db.Column(db.Numeric(10, 2), nullable=False)
    promptpay_qr_payload = db.Column(db.String(255), nullable=True)

    status = db.Column(
        SAEnum('pending', 'approved', 'paid', 'active', 'rejected', 'expired', name='order_status_enum'),
        nullable=False,
        default='pending'
    )

    slip_image = db.Column(db.String(255), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.now)  # คงรูปแบบเดิม
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    show_at = db.Column(db.Date, nullable=True)

    def __repr__(self):
        return f'<Order {self.id}>'


# โมเดล Ad (โฆษณา)
class Ad(db.Model):
    __tablename__ = 'ads'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    order_id = db.Column(db.Integer, nullable=False)

    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    link = db.Column(db.String(255), nullable=True)
    image = db.Column(db.String(255), nullable=True)

    status = db.Column(
        SAEnum('pending', 'approved', 'paid', 'active', 'rejected', 'expired', name='ad_status_enum'),
        nullable=False,
        default='pending'
    )

    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    expiration_date = db.Column(db.Date, nullable=True)

    admin_notes = db.Column(db.Text, nullable=True)
    admin_slip = db.Column(db.String(255), nullable=True)

    show_at = db.Column(db.Date, nullable=True)

    def __repr__(self):
        return f'<Ad {self.id}>'


# โมเดล AdPackage (แพ็กเกจโฆษณา)
class AdPackage(db.Model):
    __tablename__ = 'ad_packages'

    package_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Numeric(10, 2), nullable=False)
    duration_days = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<AdPackage {self.package_id}>'


# ========== Secrets/Env ==========
load_dotenv()
JWT_SECRET = os.getenv('JWT_SECRET')  # ใช้ค่าเดิม ถ้าไม่มีให้ไปตั้งใน .env
if not JWT_SECRET:
    # เตือนแบบไม่ทำให้แอปล่ม (เผื่อ dev ลืมตั้งค่า)
    print("[WARN] JWT_SECRET is not set in environment. Please configure it for production.")



# ==================== RECOMMENDATION SYSTEM FUNCTIONS ====================

# ============================ CONFIG (ปรับง่ายที่เดียว) =========================
DB_URI = os.getenv("BESTPICK_DB_URI", "mysql+mysqlconnector://root:1234@localhost/bestpick")

# ตาราง/วิว
POSTS_TABLE       = "posts"
USERS_TABLE       = "users"
LIKES_TABLE       = "likes"
EVENT_TABLE       = "user_interactions"
CONTENT_VIEW      = "contentbasedview"
FOLLOWS_TABLE     = "follower_following"
INCLUDE_SELF_POSTS_IN_FEED   = True   
USE_AUTHORED_AS_SIGNALS      = True   
AUTHORED_CATEGORY_BONUS      = 0.7   
AUTHORED_TEXT_BONUS          = 0.7     

# คอลัมน์ฟีเจอร์จาก content view
CATEGORY_COLS = [
    "Electronics_Gadgets",
    "Furniture",
    "Outdoor_Gear",
    "Beauty_Products",
    "Accessories",
]
TEXT_COL   = "Content"
ENGAGE_COL = "PostEngagement"

# น้ำหนัก action → implicit rating (Collaborative)
ACTION_WEIGHT = {
    "view": 1.0,
    "like": 2.0,
    "unlike": -1.0,
    "comment": 3.0,
    "bookmark": 4.0,
    "unbookmark": -2.0,
    "share": 5.0,
}
POS_ACTIONS      = {"view","like","comment","bookmark","share"}
NEG_ACTIONS      = {"unlike","unbookmark"}
IGNORE_ACTIONS   = {"view_profile","follow","unfollow"}
VIEW_POS_MIN     = 1
RATING_MIN, RATING_MAX = 0.5, 5.0

# Hybrid weights (อยาก “ยกหมวดหมู่” ขึ้นก็ปรับ WEIGHT_CATEGORY)
WEIGHT_COLLAB    = 0.25
WEIGHT_ITEM      = 0.20
WEIGHT_USER_TEXT = 0.20
WEIGHT_CATEGORY  = 0.30   # <<<<<<<<<<<<<< หมวดหมู่สำคัญขึ้น
WEIGHT_POP       = 0.05

# TF-IDF & item-content
TFIDF_PARAMS = dict(analyzer="char_wb", ngram_range=(2,5), max_features=60000, min_df=2, max_df=0.95)
KNN_NEIGHBORS = 10

# Popularity prior
POP_ALPHA = 5.0  # Bayesian smoothing

# Cache/TTL
OUT_DIR = "./recsys_eval_final"
CACHE_DIR = os.path.join(OUT_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

CACHE_EXPIRY_TIME_SECONDS = 120
IMPRESSION_HISTORY_TTL_SECONDS = 24*3600
IMPRESSION_HISTORY_MAX_ENTRIES = 500
INCLUDE_SELF_POSTS = False  # รวมโพสต์เจ้าของเองไหม

# ====================== DIVERSITY / NEWNESS / THRESHOLDS ======================
RUNLEN_CAP_TOP20 = 4
RUNLEN_CAP_AFTER = 4
MMR_LAMBDA       = 0.80
MMR_MAX_REF      = 30

NEW_WINDOWS_HOURS = [1, 3, 24]
NEW_INSERT_MAX    = 3

CAT_MATCH_TOP20 = 0.60
CAT_MATCH_AFTER = 0.50
ENG_PCTL_TOP20  = 40
ENG_PCTL_NEW    = 25

TEMP_UNSEEN = 0.15
TEMP_SEENNO = 0.12
TEMP_INTER  = 0.10

# สำหรับ _rank._final_score (โซน Top20/21-30 ใช้สเกลนี้)
WEIGHT_E = 0.50  # engagement
WEIGHT_C = 0.25  # category match (self)
WEIGHT_F = 0.10  # follow-influence category
WEIGHT_T = 0.10  # text relevance
WEIGHT_R = 0.05  # recency (ใช้เฉพาะโซน new)

# ---- logging to file ----
LOGREC_FILE = os.path.join(OUT_DIR, "logrec.txt")
_log_lock = threading.Lock()

# =============================== GLOBAL STATE ===================================
recommendation_cache: Dict[int, Dict] = {}
impression_history_cache: Dict[int, List[Dict]] = {}
_cache_lock = threading.Lock()

# ContentBased global (lazy-build)
_tfidf = None
_X = None
_postidx: Dict[int, int] = {}

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

def _normalize_vec(v) -> np.ndarray:
    """min-max normalize สำหรับเวกเตอร์ numpy (ใช้กับโปรไฟล์หมวด)"""
    v = np.asarray(v, dtype=np.float32)
    if v.size == 0:
        return v
    mn, mx = float(np.min(v)), float(np.max(v))
    rng = mx - mn
    if rng <= 1e-12:
        return np.zeros_like(v, dtype=np.float32)
    return (v - mn) / (rng + 1e-12)

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
def _load_posts_active(e) -> pd.DataFrame:
    """โหลดโพสต์ (กรอง active ตามคอลัมน์ที่มี) สำหรับตรวจ recency/active coverage"""
    df = pd.read_sql(f"SELECT * FROM {POSTS_TABLE}", e)
    cols = {c.lower(): c for c in df.columns}
    if "status" in cols:
        df = df[df[cols["status"]].astype(str).str.lower().isin(["active","published","publish","1","true"])]
    elif "active" in cols:
        df = df[df[cols["active"]].astype(str).str.lower().isin(["1","true","t","yes","y","active"])]
    elif "is_active" in cols:
        df = df[df[cols["is_active"]].astype(str).str.lower().isin(["1","true","t","yes","y"])]
    return df

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



def _apply_category_runlen_cap(
    ids: List[int],
    content_df: pd.DataFrame,
    user_cat_prof: np.ndarray,
    scores_by_pid: Dict[int, float],
    cap: int = 3
) -> List[int]:
    """
    เรียงใหม่โดย 'ห้าม' มีหมวดเดียวกันติดกันเกิน cap แต่ไม่บังคับให้ต้องเป็น 3 เสมอ
    เลือกตัวถัดไปแบบ greedy โดยดูคะแนนเดิม (scores_by_pid) + preference หมวดของผู้ใช้เล็กน้อย
    """
    if not ids: return []

    # เตรียม mapping pid -> category index/name
    idx = content_df.set_index("post_id")
    def _cat_idx(pid: int) -> int:
        row = idx.loc[pid, CATEGORY_COLS].to_numpy(dtype=np.float32)
        return int(np.argmax(row)) if row.size else 0
    def _cat_name(pid: int) -> str:
        return CATEGORY_COLS[_cat_idx(pid)]

    # เตรียมคิวโดยแยกตามหมวด + เรียงในหมวดตามคะแนน
    buckets: Dict[int, List[int]] = {}
    for pid in ids:
        ci = _cat_idx(pid)
        buckets.setdefault(ci, []).append(pid)
    for ci in buckets:
        buckets[ci].sort(key=lambda p: (scores_by_pid.get(p, 0.0)), reverse=True)

    # ดึงค่า preference ของผู้ใช้ต่อหมวด (ไว้ช่วย break tie ตอนคะแนนใกล้กัน)
    user_pref = (user_cat_prof if user_cat_prof is not None and user_cat_prof.size==len(CATEGORY_COLS)
                 else np.zeros(len(CATEGORY_COLS), dtype=np.float32))

    out: List[int] = []
    last_cat = None
    run_len = 0

    # สร้างชุด "ผู้สมัคร" = หัวแถวของแต่ละหมวดที่ยังเหลือ
    def _candidates(exclude_cat: Optional[int]) -> List[Tuple[float,int,int]]:
        cands = []
        for ci, lst in buckets.items():
            if not lst: continue
            if exclude_cat is not None and ci == exclude_cat and run_len >= cap:
                # cat เดิมชนเพดาน cap แล้ว → ห้าม
                continue
            head = lst[0]
            base = scores_by_pid.get(head, 0.0)
            bonus = 0.02 * float(user_pref[ci])  # เล็กน้อยพอช่วย balance แต่ไม่สั่นแรงเกิน
            cands.append((base + bonus, ci, head))
        cands.sort(key=lambda x: x[0], reverse=True)
        return cands

    total_left = sum(len(v) for v in buckets.values())
    while total_left > 0:
        # เลือกผู้สมัครที่ดีที่สุดโดยไม่ทำให้ run_len > cap
        cands = _candidates(exclude_cat=last_cat)
        if not cands:
            # ทุกตัวที่เหลือคือหมวดเดียวกับ last_cat และวิ่งชน cap หมดแล้ว
            # ยอมคลี่ constraint (fallback) เพื่อไม่ติด deadlock
            # -> เลือกคะแนนสูงสุดที่เหลือ (แม้จะทำให้ run เกิน cap ในทางทฤษฎี แต่กรณีนี้คือไม่มีทางเลือก)
            cands = _candidates(exclude_cat=None)
            if not cands:
                break

        _, ci, pid = cands[0]
        # เอาออกจาก bucket
        buckets[ci].pop(0)
        total_left -= 1

        # อัปเดต run
        if last_cat is None or ci != last_cat:
            last_cat = ci
            run_len = 1
        else:
            run_len += 1

        out.append(pid)

    # ในทางปฏิบัติ logic นี้จะไม่ “ยัด 3 เสมอ” แต่จะพยายามรักษาคะแนนรวม + cap constraint
    return out

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

    # -------- cache (ต่อ user) --------
    with _cache_lock:
        cached = recommendation_cache.get(uid)
        if use_cache and cached and (now - cached["timestamp"]).total_seconds() < CACHE_EXPIRY_TIME_SECONDS:
            return [int(x) for x in cached["ids"]]

    e = _eng()
    content_df = _load_content_view(e)
    events_all = _load_events_all(e)

    # ----- blocks / cache ไฟล์สำหรับ TF-IDF / KNN / SVD / user-text profiles -----
    content_hash = _md5_of_df(content_df, ["post_id", TEXT_COL, ENGAGE_COL])
    ev_sample = events_all[["user_id","post_id","action_type"]].head(5000) if len(events_all)>5000 else events_all[["user_id","post_id","action_type"]]
    events_hash = _md5_of_df(ev_sample, ["user_id","post_id","action_type"])
    cache_key = f"{content_hash}_{events_hash}"
    fp_tfidf = os.path.join(CACHE_DIR, cache_key + ".tfidf.pkl")
    fp_X     = os.path.join(CACHE_DIR, cache_key + ".X.npz")
    fp_knn   = os.path.join(CACHE_DIR, cache_key + ".knn.pkl")
    fp_item  = os.path.join(CACHE_DIR, cache_key + ".item.npy")
    fp_ut    = os.path.join(CACHE_DIR, cache_key + ".ut.pkl")
    fp_svd   = os.path.join(CACHE_DIR, cache_key + ".svd.pkl")
    ensure_models_built(content_df, events_all, cache_key, CACHE_DIR, force=False, non_blocking=True)


    # tfidf/X/postidx (safe load / atomic save)
    tfidf = _safe_pickle_load(fp_tfidf)
    X = None
    postidx = {pid:i for i, pid in enumerate(content_df["post_id"].astype(int).tolist())}

    if tfidf is not None and os.path.exists(fp_X):
        try:
            X = load_npz(fp_X).astype(np.float32)
        except Exception:
            try: os.remove(fp_X)
            except Exception: pass
            tfidf = None
            X = None

    if tfidf is None or X is None:
        tfidf, X, postidx, _ = build_contentbased_models(content_df)
        try:
            _atomic_write_file(fp_tfidf, pickle.dumps(tfidf))
        except Exception:
            try:
                with open(fp_tfidf + ".tmp", "wb") as f:
                    pickle.dump(tfidf, f)
                    f.flush()
                    try: os.fsync(f.fileno())
                    except Exception: pass
                os.replace(fp_tfidf + ".tmp", fp_tfidf)
            except Exception:
                pass
        try:
            save_npz(fp_X, X)
        except Exception:
            try:
                if os.path.exists(fp_X): os.remove(fp_X)
                save_npz(fp_X, X)
            except Exception:
                pass

    # KNN + item-content scores (safe load / atomic save)
    knn = _safe_pickle_load(fp_knn)
    item_scores = None
    if knn is not None and os.path.exists(fp_item):
        try:
            item_scores = np.load(fp_item, allow_pickle=False)
        except Exception:
            try: os.remove(fp_item)
            except Exception: pass
            knn = None
            item_scores = None

    if knn is None or item_scores is None:
        knn = _build_knn(X)
        item_scores = _precompute_item_content_scores(knn, content_df, X)
        try:
            _atomic_write_file(fp_knn, pickle.dumps(knn))
        except Exception:
            try:
                with open(fp_knn + ".tmp", "wb") as f:
                    pickle.dump(knn, f)
                os.replace(fp_knn + ".tmp", fp_knn)
            except Exception:
                pass
        try:
            np.save(fp_item, item_scores)
        except Exception:
            try:
                if os.path.exists(fp_item): os.remove(fp_item)
                np.save(fp_item, item_scores)
            except Exception:
                pass

    # user-text profiles (label y=1 จาก POS_ACTIONS/view) - safe load/write
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

    ut_profiles = _safe_pickle_load(fp_ut)
    if ut_profiles is None:
        ut_profiles = _user_text_profiles(train_pos, content_df, X)
        try:
            _atomic_write_file(fp_ut, pickle.dumps(ut_profiles))
        except Exception:
            try:
                with open(fp_ut + ".tmp", "wb") as f:
                    pickle.dump(ut_profiles, f)
                os.replace(fp_ut + ".tmp", fp_ut)
            except Exception:
                pass

    # collaborative SVD (safe load/write)
    svd = _safe_pickle_load(fp_svd)
    if svd is None:
        svd = build_collaborative_model(events_all, content_df["post_id"].astype(int).tolist())
        try:
            _atomic_write_file(fp_svd, pickle.dumps(svd))
        except Exception:
            try:
                with open(fp_svd + ".tmp", "wb") as f:
                    pickle.dump(svd, f)
                os.replace(fp_svd + ".tmp", fp_svd)
            except Exception:
                pass

    # -------- user-specific --------
    user_events   = events_all[events_all["user_id"] == int(uid)]
    user_cat_prof = _user_category_profile(uid, user_events, content_df)

    # hybrid scores → อันดับฐาน
    sc = compute_hybridrecommendation_scores(
        uid, content_df, tfidf, X, postidx, ut_profiles, svd, item_scores, user_cat_prof
    )
    ordered_raw = [int(x) for x in sc["post_id"].tolist()]

    # --- สร้าง mapping สำหรับความสำคัญของโพสต์ (ใช้ใน diversity อิง ranking เดิม) ---
    # ใช้ final_norm ถ้ามี ไม่งั้นใช้ final
    score_col = "final_norm" if "final_norm" in sc.columns else "final"
    scores_by_pid = {int(pid): float(s) for pid, s in zip(sc["post_id"].astype(int), sc[score_col].astype(float))}

    # กันโพสต์ที่ตัวเองเป็นคนโพสต์
    try:
        my_posts = set(_get_authored_ids(_eng(), uid))
    except Exception:
        my_posts = set()
    ordered_raw = [pid for pid in ordered_raw if pid not in my_posts]

    # กันซ้ำ preserve-order
    seen_once = set(); base_order = []
    for pid in ordered_raw:
        if pid not in seen_once:
            base_order.append(pid); seen_once.add(pid)

    # ===== ใช้ Impression history แยกเป็น unseen / seen_no_positive / interacted =====
    unseen, seen_no_pos, interacted = _split_seen_buckets(uid, base_order, events_all)

    # ===== diversity (run-length cap = 3) ต่อบล็อก + ผสานแบบเลี่ยงชนขอบ =====
    block_unseen     = _apply_category_runlen_cap(unseen,       content_df, user_cat_prof, scores_by_pid, cap=3)
    block_seen_no    = _apply_category_runlen_cap(seen_no_pos,  content_df, user_cat_prof, scores_by_pid, cap=3)
    block_interacted = _apply_category_runlen_cap(interacted,   content_df, user_cat_prof, scores_by_pid, cap=3)

    merged = _concat_with_boundary_cap(block_unseen, block_seen_no, content_df, cap=3, scores_by_pid=scores_by_pid)
    merged = _concat_with_boundary_cap(merged,       block_interacted, content_df, cap=3, scores_by_pid=scores_by_pid)

    # -------- update cache (impressions จะไป record ใน handler หลังส่งจริง) --------
    with _cache_lock:
        recommendation_cache[uid] = {"ids": merged, "timestamp": now}

    return merged


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
          unseen: List[int], seen_no: List[int], interacted: List[int]) -> List[int]:

    now = datetime.now()

    # เตรียม TF-IDF / post-index
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
        # ใช้ WEIGHT_* เดิมของคุณได้เลย (หรือ map จากชุด WEIGHT_COLLAB/ITEM/USER_TEXT/CATEGORY/POP ถ้าคุณใช้ชื่อแบบนั้น)
        if zone_is_new:
            return WEIGHT_E*E + WEIGHT_C*C + WEIGHT_F*F + WEIGHT_T*T + WEIGHT_R*R
        return WEIGHT_E*E + WEIGHT_C*C + WEIGHT_F*F + WEIGHT_T*T

    base_score = {int(pid): _final_score(scores_E[int(pid)], scores_C[int(pid)], scores_F[int(pid)],
                                         scores_T[int(pid)], 0.0, zone_is_new=False)
                  for pid in content_df["post_id"].astype(int)}

    # เตรียมข้อมูลช่วยตัดสิน
    ptiles = _category_percentiles_map(content_df)

    # set ของโพสต์ที่ตัวเองเป็นเจ้าของ (กัน/อนุญาตในฟีดตาม flag)
    try:
        self_post_ids = set(_get_authored_ids(_eng(), user_id))
    except Exception:
        self_post_ids = set()

    # ===================== Top 20 =====================
    top20_cands = []
    pool = unseen + seen_no
    for pid in pool:
        if (not INCLUDE_SELF_POSTS_IN_FEED) and (pid in self_post_ids):
            continue
        cat = category_by_pid(content_df, pid)
        ok_cat = scores_C.get(pid, 0.0) >= CAT_MATCH_TOP20
        thr_map = ptiles[str(ENG_PCTL_TOP20)]
        thr = thr_map.get(cat, thr_map["__global__"])
        ok_eng = content_df.loc[content_df["post_id"] == pid, "PostEngagement"].values[0] >= thr
        if ok_cat and ok_eng:
            top20_cands.append(pid)

    if len(top20_cands) < 20:
        relax_pool = [pid for pid in pool if pid not in top20_cands]
        relax = sorted(relax_pool, key=lambda x: (scores_C.get(x,0.0), base_score.get(x,0.0)), reverse=True)
        for pid in relax:
            if (not INCLUDE_SELF_POSTS_IN_FEED) and (pid in self_post_ids):
                continue
            if len(top20_cands) >= 20:
                break
            top20_cands.append(pid)

    # MMR diversity
    def simfunc(a:int,b:int):
        ia, ib = _postidx.get(a,-1), _postidx.get(b,-1)
        if ia<0 or ib<0: return 0.0
        va, vb = _X[ia], _X[ib]
        num = float(va.multiply(vb).sum())
        den = (np.linalg.norm(va.data)*np.linalg.norm(vb.data))
        return float(num/den) if den>0 else 0.0

    top20_sorted = _mmr_select(
        candidates=sorted(set(top20_cands), key=lambda x: base_score.get(x,0.0), reverse=True),
        scores=base_score, simfunc=simfunc, lam=MMR_LAMBDA, k=min(20, len(content_df))
    )

    top20_out, cat_seq = [], []
    for pid in top20_sorted:
        if (not INCLUDE_SELF_POSTS_IN_FEED) and (pid in self_post_ids):
            continue
        cat = category_by_pid(content_df, pid)
        if _runlen_violate(cat_seq, cat, RUNLEN_CAP_TOP20):
            continue
        top20_out.append(pid); cat_seq.append(cat)
        if len(top20_out) >= 20: break

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
            if row.empty: continue
            ts = pd.to_datetime(row["created_ts"].iloc[0]) if "created_ts" in row.columns else pd.NaT
            if not _is_new(ts, now_ts, h): continue
            cat = category_by_pid(content_df, pid)
            cond_cat = scores_C.get(pid,0.0) >= CAT_MATCH_AFTER
            thr_map_new = ptiles[str(ENG_PCTL_NEW)]
            thr_new = thr_map_new.get(cat, thr_map_new["__global__"])
            cond_eng = row["PostEngagement"].values[0] >= thr_new
            if cond_cat and cond_eng:
                sc = _final_score(scores_E.get(pid,0.0), scores_C.get(pid,0.0), scores_F.get(pid,0.0),
                                  scores_T.get(pid,0.0), scores_R.get(pid,0.0), zone_is_new=True)
                new_cands.append((sc, pid))
        new_cands.sort(reverse=True)
        picked = [pid for _,pid in new_cands[:need_max]]
        if picked:
            zone21_30 = picked[:need_max]
            break

    # เติม 21–30 ด้วย best-of-rest
    chosen20 = set(top20_out)
    chosen21 = set(zone21_30)
    remaining_pool = [pid for pid in unseen + seen_no if pid not in chosen20 | chosen21]
    if not INCLUDE_SELF_POSTS_IN_FEED:
        remaining_pool = [pid for pid in remaining_pool if pid not in self_post_ids]
    rest_sorted = sorted(remaining_pool, key=lambda x: base_score.get(x,0.0), reverse=True)

    pos21_30 = []
    insert_positions = [22, 26, 29]
    i_new = 0
    for pos in range(21, 31):
        if i_new < len(zone21_30) and pos in insert_positions:
            pos21_30.append(zone21_30[i_new]); i_new += 1
        elif rest_sorted:
            pos21_30.append(rest_sorted.pop(0))

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

    tail, cat_seq_all = [], [category_by_pid(content_df, pid) for pid in (top20_out + pos21_30)]
    pos_idx = len(top20_out) + len(pos21_30)

    # fill จนถึงจุดเริ่มแทรก interacted
    mix_pool = sorted(unseen_rest + seenno_rest, key=lambda x: base_score.get(x,0.0), reverse=True)
    for pid in mix_pool:
        cat = category_by_pid(content_df, pid)
        if _runlen_violate(cat_seq_all, cat, RUNLEN_CAP_AFTER):
            continue
        tail.append(pid); cat_seq_all.append(cat)
        pos_idx += 1
        if pos_idx >= start_inter:
            break

    remain_ids = [pid for pid in base_rest if pid not in set(tail)]
    remain_unseen   = [pid for pid in unseen_rest   if pid in remain_ids]
    remain_seenno   = [pid for pid in seenno_rest   if pid in remain_ids]
    remain_inter    = [pid for pid in interact_rest if pid in remain_ids]

    while remain_unseen or remain_seenno or remain_inter:
        block_items = []
        quota_inter = max(0, int(0.10 * 10))  # ~10% ต่อบล็อก 10
        for _ in range(10):
            best = None; best_score = -1
            pools = [("unseen", remain_unseen), ("seenno", remain_seenno), ("inter", remain_inter if quota_inter>0 else [])]
            for name, pool in pools:
                if not pool: continue
                cand = pool[0]
                s = base_score.get(cand, 0.0)
                if s > best_score:
                    best_score = s; best = (name, cand)
            if best is None: break
            name, cand = best
            cat = category_by_pid(content_df, cand)
            if _runlen_violate(cat_seq_all, cat, RUNLEN_CAP_AFTER):
                pool = remain_unseen if name=="unseen" else remain_seenno if name=="seenno" else remain_inter
                pool.pop(0)
                continue
            block_items.append(cand); cat_seq_all.append(cat)
            pool = remain_unseen if name=="unseen" else remain_seenno if name=="seenno" else remain_inter
            pool.pop(0)
            if name == "inter": quota_inter -= 1
            if len(block_items) >= 10: break

        if not block_items:
            leftovers = remain_unseen + remain_seenno + remain_inter
            for cand in leftovers:
                cat = category_by_pid(content_df, cand)
                if _runlen_violate(cat_seq_all, cat, RUNLEN_CAP_AFTER):
                    continue
                block_items.append(cand)
            remain_unseen.clear(); remain_seenno.clear(); remain_inter.clear()
        tail.extend(block_items)

    final_list = top20_out + pos21_30 + tail
    ordered, seen_set = [], set()
    for pid in final_list:
        if pid not in seen_set:
            ordered.append(pid); seen_set.add(pid)
    return ordered


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


def _interleave_balanced_with_cap(ids: List[int],
                                  content_df: pd.DataFrame,
                                  user_cat_prof: np.ndarray,
                                  scores_by_pid: Dict[int, float],
                                  cap: int = 3,
                                  alpha_pref: float = 0.5,
                                  beta_head: float = 0.35,
                                  gamma_avail: float = 0.15) -> List[int]:
    """
    จัดเรียง ids ใหม่ให้:
      - ไม่ให้หมวดเดียวติดเกิน cap (ถ้ายังมีหมวดอื่นให้สแทรก)
      - เลือกหมวดถัดไปจาก utility = alpha*pref + beta*headScore + gamma*availShare
      - รักษา order ภายในหมวด (ใช้คิวต่อหมวด)
      - ถ้าเหลือหมวดเดียวจริง ๆ -> อนุญาตให้ทะลุ cap (เลี่ยงไม่ได้)
    """

    if not ids:
        return []

    # map pid -> category
    idx = content_df.set_index("post_id")
    def _cat_of(pid: int) -> str:
        try:
            row = idx.loc[int(pid)]
            vals = row[CATEGORY_COLS].to_numpy(dtype=np.float32)
            return CATEGORY_COLS[int(np.argmax(vals))] if vals.size else "Unknown"
        except Exception:
            return "Unknown"

    # จัดคิวต่อหมวด (preserve order ภายในหมวด)
    from collections import defaultdict, deque
    queues: Dict[str, deque] = defaultdict(deque)
    for pid in ids:
        queues[_cat_of(pid)].append(pid)

    # ความชอบหมวด (normalize)
    pref_vec = np.asarray(user_cat_prof, dtype=np.float32)
    if pref_vec.size != len(CATEGORY_COLS) or float(pref_vec.sum()) <= 0:
        pref_vec = np.ones(len(CATEGORY_COLS), dtype=np.float32)
    pref_vec = pref_vec / (pref_vec.sum() + 1e-12)
    pref_map = {CATEGORY_COLS[i]: float(pref_vec[i]) for i in range(len(CATEGORY_COLS))}

    # เพิ่ม "Unknown" = ค่าต่ำสุดเล็กน้อย
    for c in list(queues.keys()):
        if c not in pref_map:
            pref_map[c] = 0.0

    # ฟังก์ชัน utility ต่อหมวด
    def _utility(cat: str, last_cat: Optional[str], run_len: int) -> float:
        if not queues[cat]:
            return -1e9
        head_pid = queues[cat][0]
        head_score = float(scores_by_pid.get(int(head_pid), 0.0))
        # สัดส่วนของโพสต์หมวดนี้ที่ยังเหลือ
        rem_c = float(len(queues[cat]))
        rem_total = float(sum(len(q) for q in queues.values()))
        avail_share = rem_c / (rem_total + 1e-12)
        # base utility
        u = (alpha_pref * float(pref_map.get(cat, 0.0))
             + beta_head * head_score
             + gamma_avail * avail_share)
        # ลงโทษถ้า cap ใกล้ชน
        if last_cat == cat and run_len >= (cap - 1):
            u -= 0.5  # penalty เบา ๆ เพื่อกระตุ้นให้สลับหมวด
        return u

    out: List[int] = []
    last_cat: Optional[str] = None
    run_len = 0

    total = sum(len(q) for q in queues.values())
    while len(out) < total:
        # หา candidate ที่ไม่ชน cap ก่อน
        cand_cats = [c for c in queues.keys() if queues[c]]
        picked = False
        best_cat, best_u = None, -1e9

        for c in cand_cats:
            # ถ้าหมวดเดียวกับก่อนหน้าและ run ชน cap แล้ว → ข้ามรอบแรก
            if last_cat == c and run_len >= cap:
                continue
            u = _utility(c, last_cat, run_len)
            if u > best_u:
                best_u = u; best_cat = c

        if best_cat is not None:
            pid = queues[best_cat].popleft()
            out.append(pid)
            if last_cat == best_cat:
                run_len += 1
            else:
                last_cat = best_cat
                run_len = 1
            picked = True

        if picked:
            continue

        # ถ้าไม่มีใครให้เลือก (เช่นเหลือหมวดเดียวจริง ๆ) → หยิบจากหมวดที่เหลือเยอะสุด
        if not picked:
            nonempty = [(c, len(queues[c])) for c in queues.keys() if queues[c]]
            if not nonempty:
                break
            nonempty.sort(key=lambda x: x[1], reverse=True)
            c = nonempty[0][0]
            pid = queues[c].popleft()
            out.append(pid)
            if last_cat == c:
                run_len += 1
            else:
                last_cat = c
                run_len = 1

    return out


def _concat_with_boundary_cap(
    ids_a: List[int],
    ids_b: List[int],
    content_df: pd.DataFrame,
    cap: int,
    scores_by_pid: Dict[int, float]
) -> List[int]:
    if not ids_a: return ids_b[:]
    if not ids_b: return ids_a[:]

    idx = content_df.set_index("post_id")
    def _cat_idx(pid: int) -> int:
        row = idx.loc[pid, CATEGORY_COLS].to_numpy(dtype=np.float32)
        return int(np.argmax(row)) if row.size else 0

    # หา run สุดท้ายของ A
    last_cat = _cat_idx(ids_a[-1])
    run_len = 1
    for i in range(len(ids_a)-2, -1, -1):
        if _cat_idx(ids_a[i]) == last_cat:
            run_len += 1
        else:
            break

    # ถ้ารันท้ายของ A ชน cap แล้ว และหัว B เป็นหมวดเดียวกัน → หาตัวคั่นจาก B
    if run_len >= cap and _cat_idx(ids_b[0]) == last_cat:
        # หาโพสต์ตัวแรกใน B ที่หมวด != last_cat ให้เลือก “ที่คะแนนรวมสูงสุด” ขึ้นมาเป็นหัว
        best_j = -1
        best_score = -1.0
        # จำกัดระยะค้นหาเพื่อไม่ทำลายลำดับมากเกินไป (เช่น มองหน้า 20 ตัวแรก)
        lookahead = min(20, len(ids_b))
        for j in range(lookahead):
            if _cat_idx(ids_b[j]) != last_cat:
                s = scores_by_pid.get(ids_b[j], 0.0)
                if s > best_score:
                    best_score = s; best_j = j
        if best_j >= 0:
            chosen = ids_b[best_j]
            rest_b = ids_b[:best_j] + ids_b[best_j+1:]
            return ids_a + [chosen] + rest_b

        # ถ้าไม่มีหมวดอื่นเลยในช่วง lookahead → ปล่อยต่อไป (เลี่ยงไม่ได้จริง ๆ)
        # (โดยรวมเรา “เคย” cap ในแต่ละบล็อกมาแล้ว โอกาสชนหนัก ๆ จึงน้อย)
    return ids_a + ids_b



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


# ==================== SLIP & PROMPTPAY FUNCTIONS (from Slip.py) ====================


# === Generate PromptPay QR for a given order (ตรวจเงื่อนไข/บันทึก payload/คืน QR base64) ===
def generate_promptpay_qr_for_order(order_id):
    order = find_order_by_id(order_id)
    if not order:
        print(f"❌ [WARN] Order ID {order_id} not found for QR generation.")
        return {"success": False, "message": "Order not found."}

    is_new_ad_approved = order["status"] == 'approved' and order.get("renew_ads_id") is None
    is_renewal_ad_pending = order["status"] == 'pending' and order.get("renew_ads_id") is not None

    if is_new_ad_approved or is_renewal_ad_pending:
        print(f"✅ [INFO] Order ID {order_id} is eligible for QR generation. Status: '{order['status']}', Renew Ad: {order.get('renew_ads_id')}.")
    else:
        log_message = f"❌ [WARN] Cannot generate QR for order {order_id}. Current status: '{order['status']}'."
        if order["status"] == 'pending' and order.get("renew_ads_id") is None:
            log_message += " (New ad order not yet approved by admin)."
            print(log_message)
            return {"success": False, "message": "Cannot generate a QR code. Please wait until an admin approves the content."}
        else:
            log_message += " (Invalid status for QR generation)."
            print(log_message)
            return {"success": False, "message": "Cannot generate a QR code. Invalid order status."}

    # ป้องกัน amount เพี้ยน/ติดลบ/เป็น None
    try:
        amount = float(order["amount"])
        if amount <= 0:
            print(f"❌ [ERROR] Invalid amount for order {order_id}: {order['amount']}")
            return {"success": False, "message": "Invalid order amount."}
    except Exception as e:
        print(f"❌ [ERROR] Amount parse failed for order {order_id}: {e}")
        return {"success": False, "message": "Invalid order amount."}

    # ไลบรารีต้องพร้อมใช้งาน
    if promptpay_qrcode is None:
        print(f"❌ [ERROR] promptpay_qrcode library not found.")
        return {"success": False, "message": "PromptPay library not found. Please install it first."}

    # ต้องมี PROMPTPAY_ID ใน env
    promptpay_id = os.getenv("PROMPTPAY_ID")
    if not promptpay_id:
        print(f"❌ [ERROR] PROMPTPAY_ID environment variable not set.")
        return {"success": False, "message": "PromptPay ID not found in settings."}

    try:
        original_scannable_payload = promptpay_qrcode.generate_payload(promptpay_id, amount)
    except Exception as e:
        print(f"❌ [ERROR] Payload generation failed for order {order_id}: {e}")
        return {"success": False, "message": "Failed to generate QR payload."}

    if not update_order_with_promptpay_payload_db(order_id, original_scannable_payload):
        print(f"❌ [ERROR] Failed to save QR Code payload to database for order {order_id}.")
        return {"success": False, "message": "Failed to save QR code data to the database."}

    try:
        print(f"✅ Generated PromptPay payload (stored in DB): {original_scannable_payload}")
        qr = qrcode.QRCode(
            version=1,
            error_correction=ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(original_scannable_payload)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buffered = io.BytesIO()
        # รองรับทั้ง qrcode รุ่นใหม่/เก่า
        (img.get_image() if hasattr(img, 'get_image') else img).save(buffered, "PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return {"success": True, "message": "QR code generated successfully.", "qrcode_base64": img_b64, "payload": original_scannable_payload}
    except Exception as e:
        print(f"❌ [ERROR] QR image encode failed for order {order_id}: {e}")
        return {"success": False, "message": "Failed to render QR image."}

# === Slip upload permission check (สถานะต้องพร้อมก่อนถึงจะอัปสลิปได้) ===
def can_upload_slip(order):
    if not order:
        return False
    is_new_ad_approved = order["status"] == 'approved' and order.get("renew_ads_id") is None
    is_renewal_ad_pending = order["status"] == 'pending' and order.get("renew_ads_id") is not None
    return is_new_ad_approved or is_renewal_ad_pending

# === Thai date formatter (รองรับ str, date, datetime) ===
def format_thai_date(date_obj):
    if not isinstance(date_obj, (datetime, date, str)):
        return "Incorrect Format"

    if isinstance(date_obj, str):
        try:
            date_obj = datetime.fromisoformat(date_obj)
        except ValueError:
            try:
                date_obj = datetime.strptime(date_obj, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    date_obj = datetime.strptime(date_obj, '%Y-%m-%d').date()
                except ValueError:
                    return "Incorrect Format"

    if isinstance(date_obj, date) and not isinstance(date_obj, datetime):
        date_obj = datetime(date_obj.year, date_obj.month, date_obj.day)

    thai_year = date_obj.year + 543
    try:
        formatted_date = date_obj.strftime(f'%d %B {thai_year}')
    except ValueError:
        thai_month_names = [
            "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
            "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"
        ]
        formatted_date = f"{date_obj.day} {thai_month_names[date_obj.month - 1]} {thai_year}"
    return formatted_date

# === Notification writer: บันทึกการเปลี่ยนสถานะโฆษณาเป็น notification ===
def notify_ads_status_change(db, ad_id: int, new_status: str, admin_notes: str = None, duration_days_from_renewal: int = None) -> bool:
    try:
        ads_query = text('SELECT user_id, expiration_date FROM ads WHERE id = :ad_id')
        ads_result = db.session.execute(ads_query, {'ad_id': ad_id}).fetchone()
        if not ads_result:
            print(f"❌ [WARN] notify_ads_status_change: Ad ID {ad_id} not found in 'ads' table.")
            return False

        user_id = ads_result[0]
        expiration_date = ads_result[1]

        content = ''
        duration_from_package_db = None

        # หาวัน duration จาก order ล่าสุดที่จ่าย/รออนุมัติการจ่าย
        if duration_days_from_renewal is None:
            order_package_query = text("""
                SELECT ap.duration_days, o.status as order_status
                FROM orders o
                JOIN ad_packages ap ON o.package_id = ap.package_id
                WHERE o.renew_ads_id = :ad_id
                  AND o.status IN ('paid', 'approved_payment_waiting')
                ORDER BY o.created_at DESC
                LIMIT 1
            """)
            package_info_result = db.session.execute(order_package_query, {'ad_id': ad_id}).fetchone()
            if package_info_result:
                duration_from_package_db = package_info_result[0]
                order_status_found = package_info_result[1]
                print(f"✅ [INFO] Found package duration {duration_from_package_db} days from order for Ad ID {ad_id} with order status '{order_status_found}'.")
            else:
                print(f"⚠️ [WARN] No 'paid' or 'approved_payment_waiting' order found linked to Ad ID {ad_id} via renew_ads_id to determine package duration.")

        if new_status == 'active':
            duration_to_use = duration_days_from_renewal if duration_days_from_renewal is not None else duration_from_package_db
            if duration_to_use is not None:
                formatted_expiration_date = format_thai_date(expiration_date)
                content = f"Your ad has been successfully renewed for {duration_to_use} days. This ad's expiration date is extended to {formatted_expiration_date}"
            else:
                content = 'Your ad has been approved for display'
        elif new_status == 'paid':
            content = 'Your ad payment has been completed. Waiting for admin review'
        elif new_status == 'approved':
            content = 'Your ad has been reviewed. Please transfer payment to display it'
        elif new_status == 'rejected':
            content = f'Your ad was rejected. Reason: {admin_notes or "-"}'
        elif new_status == 'expired':
            content = 'Your ad has expired'
        elif new_status == 'expiring_soon':
            content = 'Your ad will expire in 3 days. Please renew to ensure continuous display'
        else:
            content = f'Your ad status has changed to {new_status}'

        insert_notification_query = text("""
            INSERT INTO notifications (user_id, action_type, content, ads_id)
            VALUES (:user_id, 'ads_status_change', :content, :ads_id)
        """)
        db.session.execute(insert_notification_query, {
            'user_id': user_id,
            'content': content,
            'ads_id': ad_id
        })
        db.session.commit()
        print(f"✅ [INFO] Notification saved successfully for Ad ID {ad_id}. Content: '{content}'")
        return True

    except Exception as e:
        print(f"❌ [ERROR] An unexpected error occurred in notify_ads_status_change for Ad ID {ad_id}: {e}")
        db.session.rollback()
        return False

# === Slip verify + update order/ad: เรียก SlipOK, ตรวจผล, อัปเดตสถานะ, แจ้งเตือน ===
def verify_payment_and_update_status(order_id, slip_image_path, payload_from_client, db):
    """
    Call SlipOK and update our order/ad. Return concise EN messages for client.
    """
    print(f"[INFO] process order={order_id}")

    order = find_order_by_id(order_id)
    if not order:
        return {"success": False, "message": "Order not found."}
    if not can_upload_slip(order):
        return {"success": False, "message": "Slip upload not allowed for this order status."}
    if not os.path.exists(slip_image_path):
        return {"success": False, "message": "Slip file missing on server."}

    # เลือกโฆษณาที่เกี่ยวข้อง (งานต่ออายุหรือสร้างใหม่)
    ad_related = None
    if order.get("renew_ads_id") is not None:
        ad_related = find_ad_by_id(order["renew_ads_id"])
        if not ad_related:
            return {"success": False, "message": "Ad for renewal not found."}
    else:
        already = find_ad_by_order_id(order_id)
        if already and already.get('status') in ['active']:
            return {"success": False, "message": "This order was already processed."}

    # SlipOK config
    SLIP_OK_API_ENDPOINT = os.getenv("SLIP_OK_API_ENDPOINT", "https://api.slipok.com/api/line/apikey/49130")
    SLIP_OK_API_KEY = os.getenv("SLIP_OK_API_KEY", "SLIPOKKBE52WN")  # ตั้ง env ให้เรียบร้อยในโปรดักชัน

    # เรียก SlipOK
    try:
        with open(slip_image_path, 'rb') as img_file:
            files = {'files': (os.path.basename(slip_image_path), img_file, 'image/jpeg')}
            data = {'log': 'true', 'amount': str(float(order["amount"]))}
            headers = {'x-authorization': SLIP_OK_API_KEY}
            r = requests.post(SLIP_OK_API_ENDPOINT, files=files, data=data, headers=headers, timeout=30)
            text_body = r.text
            r.raise_for_status()
            resp = r.json()
    except requests.exceptions.Timeout:
        print(f"[ERR] SlipOK timeout order={order_id}")
        return {"success": False, "message": "Verification service timeout. Please try again."}
    except requests.exceptions.HTTPError as e:
        msg = "Verification failed."
        try:
            j = r.json()
            code = j.get('code')
            slipok_msg = j.get('message', '')
            if code == 1002:
                msg = "Verification provider rejected our credentials. Contact support."
            elif code == 1012:
                msg = "Duplicate slip. This slip was already submitted."
            else:
                msg = slipok_msg or msg
        except Exception:
            msg = f"Verification failed: HTTP {r.status_code}"
        print(f"[ERR] SlipOK HTTP {r.status_code} order={order_id} msg={msg} body={text_body[:200]}")
        return {"success": False, "message": msg}
    except requests.exceptions.RequestException as e:
        print(f"[ERR] SlipOK connect error order={order_id}: {e}")
        return {"success": False, "message": "Cannot reach verification service. Please try again."}
    except Exception as e:
        print(f"[ERR] SlipOK call unexpected order={order_id}: {e}")
        return {"success": False, "message": "Unexpected error during verification."}

    # ตรวจผล SlipOK
    if not resp.get("success"):
        msg = resp.get("message") or "Verification failed."
        code = resp.get("code")
        if code == 1012:
            msg = "Duplicate slip. This slip was already submitted."
        return {"success": False, "message": msg}

    data = resp.get("data") or {}
    slip_amount = float(data.get("amount", 0))
    trans_ref = data.get("transRef")
    if not trans_ref:
        return {"success": False, "message": "Slip verification returned no transaction ID."}

    if abs(slip_amount - float(order.get("amount"))) > 0.01:
        return {"success": False, "message": f"Incorrect amount. Expected {order.get('amount'):.2f}, got {slip_amount:.2f}."}

    # อัปเดตระบบของเรา (order/ad/notification)
    try:
        ok = update_status_and_slip_info(order_id, "paid", slip_image_path, trans_ref)
        if not ok:
            raise RuntimeError("Update order failed")

        ad_id_to_return = None
        if order.get("renew_ads_id") is not None:
            # ต่ออายุ: คำนวณวันหมดอายุใหม่
            duration_days = get_ad_package_duration(order["package_id"])
            if duration_days is None:
                raise RuntimeError("Missing package duration")
            current_ad = find_ad_by_id(order["renew_ads_id"])
            start = current_ad.get('expiration_date') or datetime.now().date()
            if isinstance(start, datetime):
                start = start.date()
            new_exp = datetime.combine(start, datetime.min.time()).date() + timedelta(days=duration_days)
            if not update_ad_for_renewal(current_ad['id'], "active", new_exp):
                raise RuntimeError("Renewal update failed")
            ad_id_to_return = current_ad['id']
            notify_ads_status_change(db, ad_id_to_return, 'active')
            message = f"Payment verified. Your ad was renewed for {duration_days} days."
        else:
            ad = find_ad_by_order_id(order_id)
            if ad:
                if not update_ad_status(ad['id'], "paid"):
                    raise RuntimeError("Ad status update failed")
                ad_id_to_return = ad['id']
            else:
                new_id = create_advertisement_db(order)
                if not new_id:
                    raise RuntimeError("Ad creation failed")
                ad_id_to_return = new_id
            notify_ads_status_change(db, ad_id_to_return, 'paid')
            message = "Payment verified. Please wait for admin to activate the ad."

        db.session.commit()
        print(f"[INFO] order={order_id} verified, ad_id={ad_id_to_return}, transRef={trans_ref}")
        return {"success": True, "message": message, "ad_id": ad_id_to_return}

    except Exception as e:
        try:
            if db and hasattr(db, 'session'):
                db.session.rollback()
        except Exception as rb:
            print(f"[WARN] rollback error: {rb}")
        print(f"[ERR] post-verify update failed order={order_id}: {e}")
        return {"success": False, "message": "Internal update failed after verification."}

# === Expiring-soon notifier: ใส่ล็อกกันรันซ้อน/ห้ามแจ้งซ้ำภายในวันเดียว ===
def check_ads_expiring_soon(db, today_date: date = None):
    if today_date is None:
        today_date = date.today()
    target_date = today_date + timedelta(days=3)

    lockname = f"expiring_run_{today_date.isoformat()}"
    conn = db.engine.raw_connection()
    try:
        cur = conn.cursor()
        # ล็อกทั้งรอบ ป้องกันเรียกซ้อน
        cur.execute("SELECT GET_LOCK(%s, 5)", (lockname,))
        got = cur.fetchone()[0]
        if not got:
            print(f"[INFO] skip: another run holds lock {lockname}")
            return 0, []

        # กันซ้ำต่อวันด้วย anti-join ที่อิงวันที่ของ DB เอง (CURDATE())
        sql = """
        INSERT INTO notifications (user_id, action_type, content, ads_id, created_at, read_status)
        SELECT a.user_id,
               'ads_status_change',
               'Your ad will expire in 3 days. Please renew to ensure continuous display',
               a.id,
               NOW(),
               0
          FROM ads a
     LEFT JOIN notifications n
            ON n.ads_id = a.id
           AND n.action_type = 'ads_status_change'
           AND n.content = 'Your ad will expire in 3 days. Please renew to ensure continuous display'
           AND DATE(n.created_at) = CURDATE()
         WHERE DATE(a.expiration_date) = %s
           AND a.status IN ('active')
           AND n.id IS NULL
        """
        cur.execute(sql, (target_date.isoformat(),))
        inserted = cur.rowcount or 0
        conn.commit()

        if inserted == 0:
            print("[INFO] no eligible ads (already notified today or none match)")
        else:
            print(f"[INFO] notified {inserted} ad(s) for expiring_soon")

        return inserted, []
    except Exception as e:
        conn.rollback()
        print("[ERR] expiring_soon atomic insert:", e)
        return 0, []
    finally:
        try:
            cur.execute("SELECT RELEASE_LOCK(%s)", (lockname,))
            conn.commit()
        except Exception:
            conn.rollback()
        try: cur.close()
        except: pass
        try: conn.close()
        except: pass

# === Scheduler: รันเช็กหมดอายุทุกเที่ยงคืนโซนเวลาไทย พร้อม app context ===
def start_expiry_scheduler(app, db):
    tz = pytz.timezone('Asia/Bangkok')
    sched = BackgroundScheduler(timezone=tz)

    def job():
        with app.app_context():
            inserted, _ = check_ads_expiring_soon(db)
            if inserted == 0:
                print("[CRON] no eligible ads (already notified today or none match)")
            else:
                print(f"[CRON] notified {inserted} ad(s) for expiring_soon")

    sched.add_job(job, 'cron', hour=0, minute=0)
    sched.start()
    print("⏰ Expiry scheduler started (runs 00:00 Asia/Bangkok)")



# ==================== FLASK ROUTES ====================
ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
ALLOWED_VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv'}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB ต่อไฟล์ (เปลี่ยนได้โดยไม่กระทบ API)

def _ext_ok(name, allowed):
    return os.path.splitext(name)[1].lower() in allowed

def _unique_name(filename: str) -> str:
    name = secure_filename(filename)
    base, ext = os.path.splitext(name)
    token = secrets.token_hex(4)
    return f"{int(time.time())}_{token}{ext.lower()}"

def _allowed_ext(filename: str, allowed: set) -> bool:
    return os.path.splitext(filename)[1].lower() in allowed

def _save_upload(file, allowed_exts: set, folder: str) -> str:
    if not file or not file.filename:
        raise ValueError("empty_file")
    if hasattr(file, 'content_length') and file.content_length and file.content_length > MAX_UPLOAD_BYTES:
        raise ValueError("file_too_large")
    if not _allowed_ext(file.filename, allowed_exts):
        raise ValueError("unsupported_extension")
    fname = _unique_name(file.filename)
    path = os.path.join(folder, fname)
    file.save(path)
    return fname, path

def _now_th():
    try:
        return datetime.now(_TH_TZ)
    except Exception:
        # fallback: manual +7
        return datetime.utcnow() + timedelta(hours=7)

def _fmt_th(dt: datetime) -> str:
    # 2025-08-21 16:19:57 (ICT)
    return dt.strftime("%Y-%m-%d %H:%M:%S") + " ICT"

# สร้างไฟล์ใหม่ “ต่อรอบการรัน” ของ AppAI.py
SESSION_LOCAL_START = _now_th()
SESSION_STAMP = SESSION_LOCAL_START.strftime("%Y%m%d_%H%M%S")
LOGREC_DIR = OUT_DIR
os.makedirs(LOGREC_DIR, exist_ok=True)
LOGREC_FILE = os.path.join(LOGREC_DIR, f"logrec_{SESSION_STAMP}_TH.txt")

_log_lock = threading.Lock()

def _append_rec_log(lines: List[str], fp: str = None):
    path = fp or LOGREC_FILE
    try:
        with _log_lock:
            with open(path, "a", encoding="utf-8") as f:
                for ln in lines:
                    f.write(ln.rstrip("\n") + "\n")
    except Exception:
        pass

# เขียน session header เมื่อเริ่มรันไฟล์
try:
    _append_rec_log([f"===== recsys session started at {_fmt_th(SESSION_LOCAL_START)} ====="])
except Exception:
    pass

# ==================== Recommendation Route (optimized, production) ====================

@app.route('/ai/recommend', methods=['POST'])
@verify_token   # ถ้าไม่ใช้ JWT ก็ลบบรรทัดนี้ออกได้
def ai_recommend():
    return ai_recommend_handler()

@app.route('/ai/seen', methods=['POST'])
@verify_token
def ai_seen():
    return ai_seen_handler()

from datetime import datetime, timedelta
import threading
from sqlalchemy import text
import traceback
import pytz

@app.route('/ai/posts/create', methods=['POST'])
def create_post():
    try:
        user_id = request.form.get('user_id')
        content = request.form.get('content', '')
        category = request.form.get('category')
        title = request.form.get('Title')
        product_name = request.form.get('ProductName')
        photos = request.files.getlist('photo')
        videos = request.files.getlist('video')

        if not user_id:
            return jsonify({"error": "You are not authorized to create a post for this user"}), 403

        photo_urls, invalid_photos = [], []
        print(f"Processing {len(photos)} photos...")

        for photo in photos:
            if not photo or not getattr(photo, 'filename', None):
                continue
            photo_path = None
            try:
                fname, photo_path = _save_upload(photo, ALLOWED_IMAGE_EXTS, UPLOAD_FOLDER)
                print(f"Processing photo: {fname}")
                is_nude, result = nude_predict_image(photo_path)

                if is_nude:
                    print(f"NSFW detected in {fname}")
                    if os.path.exists(photo_path):
                        os.remove(photo_path)
                    invalid_photos.append({
                        "filename": fname,
                        "reason": "พบภาพโป๊ (Hentai หรือ Pornography > 20%)",
                        "details": result
                    })
                else:
                    print(f"Photo {fname} is safe")
                    photo_urls.append(f'/uploads/{fname}')
            except Exception:
                print(f"Error processing photo {getattr(photo, 'filename', '?')}:")
                traceback.print_exc()
                if photo_path and os.path.exists(photo_path):
                    try:
                        os.remove(photo_path)
                    except Exception:
                        pass
                invalid_photos.append({
                    "filename": getattr(photo, 'filename', '?'),
                    "reason": "Unable to process the image.",
                    "details": {"error": "processing error (see log)"}
                })

        print(f"Invalid photos found: {len(invalid_photos)}")
        print(f"Valid photos: {len(photo_urls)}")

        if invalid_photos:
            print("แจ้งเตือน user: พบภาพไม่เหมาะสม")
            return jsonify({
                "status": "warning",
                "message": "กรุณาเปลี่ยนภาพแล้วลองใหม่อีกครั้ง",
                "invalid_photos": invalid_photos,
                "valid_photos": photo_urls
            }), 400

        video_urls = []
        for video in videos:
            if not video or not getattr(video, 'filename', None):
                continue
            try:
                vname, vpath = _save_upload(video, ALLOWED_VIDEO_EXTS, UPLOAD_FOLDER)
                video_urls.append(f'/uploads/{vname}')
            except Exception:
                print(f"Skip invalid video {getattr(video, 'filename', '?')}:")
                traceback.print_exc()
                # ไม่ fail ทั้งโพสต์เพราะวิดีโอพัง

        photo_urls_json = json.dumps(photo_urls)
        video_urls_json = json.dumps(video_urls)

        # --- ใช้เวลาเป็น Asia/Bangkok แล้วส่งเข้า DB ใน INSERT แบบเป็นค่าสตริง (ไม่แก้ schema) ---
        try:
            bkk_tz = pytz.timezone('Asia/Bangkok')
            now_bkk = datetime.now(bkk_tz)
            created_at_str = now_bkk.strftime('%Y-%m-%d %H:%M:%S')

            insert_query = text("""
                INSERT INTO posts (
                    user_id, Title, content, ProductName, CategoryID,
                    photo_url, video_url, status, created_at, updated_at
                )
                VALUES (
                    :user_id, :title, :content, :product_name, :category_id,
                    :photo_urls, :video_urls, 'active', :created_at, :updated_at
                )
            """)
            result = db.session.execute(insert_query, {
                'user_id': user_id,
                'title': title,
                'content': content,
                'product_name': product_name,
                'category_id': category,
                'photo_urls': photo_urls_json,
                'video_urls': video_urls_json,
                'created_at': created_at_str,
                'updated_at': created_at_str
            })
            db.session.commit()

            post_id = getattr(result, "lastrowid", None)
            if not post_id:
                try:
                    post_id = db.session.execute(text("SELECT LAST_INSERT_ID()")).scalar()
                except Exception:
                    post_id = None

            print(f"Post created successfully with ID: {post_id}, {len(photo_urls)} photos and {len(video_urls)} videos")

        except Exception:
            print("Database error during insert:")
            traceback.print_exc()
            db.session.rollback()
            for url in photo_urls:
                try:
                    path = os.path.join(UPLOAD_FOLDER, os.path.basename(url))
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
            for url in video_urls:
                try:
                    path = os.path.join(UPLOAD_FOLDER, os.path.basename(url))
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
            return jsonify({"error": "ไม่สามารถบันทึกโพสต์ลงฐานข้อมูลได้"}), 500

        # ---- notification: พยายามเรียกฟังก์ชันเดิม ถ้าไม่มี หรือโยน error ให้ insert notification
        # ---- สำหรับทุกคนที่ follow ผู้โพสต์ (จากตาราง follower_following)
        try:
            def _notify_in_app_context(_post_id, _poster_user_id, _content_snippet):
                try:
                    with app.app_context():
                        # ถ้ามีฟังก์ชัน create_notifications_for_post ให้เรียกมันก่อน (ถ้ามี)
                        notif_func = globals().get('create_notifications_for_post')
                        external_called = False
                        if callable(notif_func):
                            try:
                                notif_func(_post_id, _poster_user_id, content_snippet=_content_snippet, run_async=False)
                                print(f"[create_post] Called external create_notifications_for_post for post {_post_id}")
                                external_called = True
                            except Exception:
                                print(f"[create_post] external create_notifications_for_post threw (ignored):")
                                traceback.print_exc()

                        # ----- regardless of external_called, ensure notifications exist for each follower -----
                        try:
                            # ดึง follower_id ทั้งหมดที่ follow poster
                            q_followers = text("""
                                SELECT follower_id
                                FROM follower_following
                                WHERE following_id = :poster_id
                            """)
                            res = db.session.execute(q_followers, {'poster_id': _poster_user_id})
                            # Try mappings() -> list of dicts; fallback to tuples
                            try:
                                followers = [r.get('follower_id') for r in res.mappings().all()]
                            except Exception:
                                keys = res.keys()
                                fetched = res.fetchall()
                                followers = [row[0] for row in fetched]  # follower_id should be first column

                            if not followers:
                                print(f"[create_post] No followers found for poster {_poster_user_id} (no notifications inserted).")
                            else:
                                print(f"[create_post] Found {len(followers)} follower(s) for poster {_poster_user_id}. Inserting notifications...")

                                # Prepare content text (english format requested)
                                notif_content = f"User {_poster_user_id} performed action: post on post {_post_id}"

                                # Insert one notification row per follower (single transaction)
                                inserted_count = 0
                                for fid in followers:
                                    try:
                                        created_at_inner = datetime.now(pytz.timezone('Asia/Bangkok')).strftime('%Y-%m-%d %H:%M:%S')
                                        insert_notif = text("""
                                            INSERT INTO notifications (user_id, post_id, action_type, content, created_at, read_status)
                                            VALUES (:notify_user_id, :post_id, 'post', :content, :created_at, 0)
                                        """)
                                        db.session.execute(insert_notif, {
                                            'notify_user_id': fid,
                                            'post_id': _post_id,
                                            'content': (notif_content or '')[:1000],
                                            'created_at': created_at_inner
                                        })
                                        inserted_count += 1
                                    except Exception:
                                        print(f"[create_post] failed to insert notification for follower {fid}:")
                                        traceback.print_exc()
                                try:
                                    db.session.commit()
                                    print(f"[create_post] Inserted {inserted_count} notification(s) for post {_post_id}")
                                except Exception:
                                    print("[create_post] commit failed after inserting notifications:")
                                    traceback.print_exc()
                                    db.session.rollback()
                        except Exception:
                            print(f"[create_post] failed while preparing/inserting follower notifications:")
                            traceback.print_exc()

                        # --- query back up to 10 notifications for this post to print a confirmation ---
                        try:
                            q = text("""
                                SELECT id, user_id, post_id, action_type, content, created_at
                                FROM notifications
                                WHERE post_id = :pid
                                ORDER BY created_at DESC
                                LIMIT 10
                            """)
                            result_proxy = db.session.execute(q, {'pid': _post_id})
                            try:
                                rows = result_proxy.mappings().all()
                            except Exception:
                                keys = result_proxy.keys()
                                fetched = result_proxy.fetchall()
                                rows = [dict(zip(keys, row)) for row in fetched]

                            if rows:
                                print(f"[create_post] Notifications for post_id={_post_id} (latest up to {len(rows)}):")
                                for r in rows:
                                    print(" - id={id}, user_id={user_id}, post_id={post_id}, action_type={action_type}, created_at={created_at}, content={content}"
                                          .format(
                                              id=r.get('id'),
                                              user_id=r.get('user_id'),
                                              post_id=r.get('post_id'),
                                              action_type=r.get('action_type'),
                                              created_at=r.get('created_at'),
                                              content=(r.get('content') or '')[:200]
                                          ))
                            else:
                                print(f"[create_post] No notifications found for post_id={_post_id} after notify attempt.")
                        except Exception:
                            print(f"[create_post] Failed to query notifications for post_id={_post_id}:")
                            traceback.print_exc()

                except Exception:
                    print(f"[create_post] notification wrapper unexpected error:")
                    traceback.print_exc()

            t = threading.Thread(target=_notify_in_app_context, args=(post_id, user_id, content), daemon=True)
            t.start()
            print(f"[create_post] Spawned notify thread for post_id={post_id}")
        except Exception:
            print("[create_post] failed to spawn notify thread:")
            traceback.print_exc()

        # ส่ง response (รวม created_at ที่เป็นเวลา Asia/Bangkok เพื่อให้ client เห็น)
        try:
            return jsonify({
                "post_id": post_id,
                "user_id": user_id,
                "content": content,
                "category": category,
                "Title": title,
                "ProductName": product_name,
                "video_urls": video_urls,
                "photo_urls": photo_urls,
                "created_at": created_at_str
            }), 201
        except Exception:
            print("Failed building response:")
            traceback.print_exc()
            return jsonify({"error": "Internal server error"}), 500

    except Exception:
        print("Internal server error in create_post:")
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500



# === Update post: ผสานไฟล์เดิม + ใหม่ ตรวจ NSFW เฉพาะรูปใหม่ ===
@app.route('/ai/posts/<int:id>', methods=['PUT'])
def update_post(id):
    try:
        Title = request.form.get('Title')
        content = request.form.get('content')
        ProductName = request.form.get('ProductName')
        CategoryID = request.form.get('CategoryID')
        user_id = request.form.get('user_id')

        existing_photos_str = request.form.get('existing_photos')
        existing_videos_str = request.form.get('existing_videos')

        try:
            existing_photos = json.loads(existing_photos_str) if existing_photos_str else []
        except Exception:
            existing_photos = []
        try:
            existing_videos = json.loads(existing_videos_str) if existing_videos_str else []
        except Exception:
            existing_videos = []

        photos = request.files.getlist('photo')
        videos = request.files.getlist('video')

        if not user_id:
            return jsonify({"error": "You are not authorized to update this post"}), 403

        photo_urls = list(existing_photos) if existing_photos else []
        invalid_photos = []

        for photo in photos:
            photo_path = None
            if not photo or not photo.filename:
                continue
            try:
                fname, photo_path = _save_upload(photo, ALLOWED_IMAGE_EXTS, UPLOAD_FOLDER)
                print(f"Processing new photo for update: {fname}")
                is_nude, result = nude_predict_image(photo_path)
                if is_nude:
                    print(f"NSFW detected in {fname} during update")
                    if os.path.exists(photo_path):
                        os.remove(photo_path)
                    invalid_photos.append({
                        "filename": fname,
                        "reason": "พบภาพโป๊ (Hentai หรือ Pornography > 20%)",
                        "details": result
                    })
                else:
                    print(f"New photo {fname} is safe")
                    photo_urls.append(f'/uploads/{fname}')
            except Exception as e:
                print(f"Error processing photo {getattr(photo, 'filename', '?')} during update: {e}")
                if photo_path and os.path.exists(photo_path):
                    os.remove(photo_path)
                invalid_photos.append({
                    "filename": getattr(photo, 'filename', '?'),
                    "reason": "Unable to process the image.",
                    "details": {"error": str(e)}
                })

        if invalid_photos:
            print("แจ้งเตือน user: พบภาพไม่เหมาะสมในระหว่างการอัปเดต")
            return jsonify({
                "status": "warning",
                "message": "กรุณาเปลี่ยนภาพแล้วลองใหม่อีกครั้ง",
                "invalid_photos": invalid_photos,
                "valid_photos": photo_urls
            }), 400

        video_urls = list(existing_videos) if existing_videos else []
        for video in videos:
            if not video or not video.filename:
                continue
            try:
                vname, vpath = _save_upload(video, ALLOWED_VIDEO_EXTS, UPLOAD_FOLDER)
                video_urls.append(f'/uploads/{vname}')
            except Exception as e:
                print(f"Skip invalid video {getattr(video, 'filename', '?')} during update: {e}")

        photo_urls_json = json.dumps(photo_urls)
        video_urls_json = json.dumps(video_urls)

        try:
            update_query = text("""
                UPDATE posts 
                SET Title = :title, content = :content, ProductName = :product_name, 
                    CategoryID = :category_id, photo_url = :photo_urls, video_url = :video_urls, 
                    updated_at = NOW()
                WHERE id = :post_id AND user_id = :user_id
            """)
            result = db.session.execute(update_query, {
                'post_id': id,
                'user_id': user_id,
                'title': Title,
                'content': content,
                'product_name': ProductName,
                'category_id': CategoryID,
                'photo_urls': photo_urls_json,
                'video_urls': video_urls_json
            })
            db.session.commit()

            if result.rowcount == 0:
                return jsonify({"error": "ไม่พบโพสต์หรือไม่มีสิทธิ์อัปเดต"}), 404

            print(f"Post updated successfully with ID: {id}")
            return jsonify({
                "post_id": id,
                "Title": Title,
                "content": content,
                "ProductName": ProductName,
                "CategoryID": CategoryID,
                "video_urls": video_urls,
                "photo_urls": photo_urls
            }), 200

        except Exception as db_error:
            print(f"Database error: {db_error}")
            db.session.rollback()
            # ไม่ลบไฟล์เก่า แต่ลบเฉพาะไฟล์ใหม่ที่เพิ่งเพิ่มในรอบนี้
            new_files = [p for p in photo_urls if p not in (existing_photos or [])]
            for url in new_files:
                try:
                    path = os.path.join(UPLOAD_FOLDER, os.path.basename(url))
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
            new_videos = [v for v in video_urls if v not in (existing_videos or [])]
            for url in new_videos:
                try:
                    path = os.path.join(UPLOAD_FOLDER, os.path.basename(url))
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
            return jsonify({"error": "ไม่สามารถอัปเดตโพสต์ลงฐานข้อมูลได้"}), 500

    except Exception as error:
        print("Internal server error:", str(error))
        return jsonify({"error": "Internal server error"}), 500


# === Update user profile: ตรวจฟิลด์บังคับ, แปลงวันเกิดหลายรูปแบบ, ตรวจ NSFW สำหรับโปรไฟล์ ===
@app.route("/ai/users/<int:userId>/profile", methods=['PUT'])
def update_user_profile(userId):
    try:
        username = request.form.get('username')
        bio = request.form.get('bio')
        gender = request.form.get('gender')
        birthday_str = request.form.get('birthday')
        profile_image_file = request.files.get('profileImage')

        if not all([username, bio, gender, birthday_str]):
            return jsonify({
                "error": "Please fill out all required fields: Username, Bio, Gender, and Date of Birth."
            }), 400

        # parse วันเกิดหลายรูปแบบ
        birthday = None
        date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"]
        try:
            birthday = datetime.fromisoformat(birthday_str).strftime("%Y-%m-%d")
        except ValueError:
            for fmt in date_formats:
                try:
                    birthday = datetime.strptime(birthday_str, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
        if birthday is None:
            return jsonify({"error": "Invalid date of birth format. Please use the format: **YYYY-MM-DD**. "}), 400

        profile_image_path = None
        if profile_image_file and profile_image_file.filename:
            temp_image_path = None
            try:
                fname, temp_image_path = _save_upload(profile_image_file, ALLOWED_IMAGE_EXTS, UPLOAD_FOLDER)
                print(f"Processing new profile image: {fname}")
                is_nude, result = nude_predict_image(temp_image_path)
                if is_nude:
                    print(f"NSFW detected in profile image {fname}")
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
                    return jsonify({
                        "status": "warning",
                        "message": "กรุณาเปลี่ยนภาพแล้วลองใหม่อีกครั้ง",
                        "details": result
                    }), 400
                else:
                    print(f"Profile image {fname} is safe")
                    profile_image_path = f'/uploads/{fname}'
            except Exception as e:
                print(f"Error processing profile image {getattr(profile_image_file, 'filename', '?')}: {e}")
                if temp_image_path and os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                return jsonify({
                    "error": "ไม่สามารถประมวลผลภาพโปรไฟล์ได้",
                    "details": str(e)
                }), 500

        # ตรวจ username ซ้ำ ยกเว้นตัวเอง
        check_username_query = text("SELECT id FROM users WHERE username = :username AND id != :user_id")
        check_results = db.session.execute(check_username_query, {
            'username': username,
            'user_id': userId
        }).fetchall()
        if len(check_results) > 0:
            return jsonify({"error": "ชื่อผู้ใช้นี้มีคนใช้แล้ว"}), 400

        # สร้าง query อัปเดต
        update_profile_query = "UPDATE users SET username = :username, bio = :bio, gender = :gender, birthday = :birthday"
        update_data = {'username': username, 'bio': bio, 'gender': gender, 'birthday': birthday, 'user_id': userId}
        if profile_image_path:
            update_profile_query += ", picture = :picture"
            update_data['picture'] = profile_image_path
        update_profile_query += " WHERE id = :user_id"

        try:
            result = db.session.execute(text(update_profile_query), update_data)
            db.session.commit()

            if result.rowcount == 0:
                return jsonify({"error": "ไม่พบผู้ใช้หรือไม่มีการเปลี่ยนแปลงข้อมูล"}), 404

            return jsonify({
                "message": "อัปเดตโปรไฟล์สำเร็จ",
                "profileImage": profile_image_path if profile_image_path else "No new image uploaded",
                "username": username,
                "bio": bio,
                "gender": gender,
                "birthday": birthday
            }), 200

        except Exception as db_error:
            print(f"Database error during profile update: {db_error}")
            db.session.rollback()
            # ไม่ลบรูปเก่าที่ผู้ใช้ใช้อยู่ ลบเฉพาะไฟล์ใหม่ถ้าเพิ่งเซฟ
            if profile_image_path:
                try:
                    p = os.path.join(UPLOAD_FOLDER, os.path.basename(profile_image_path))
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            return jsonify({"error": "เกิดข้อผิดพลาดฐานข้อมูลขณะอัปเดตโปรไฟล์ผู้ใช้"}), 500

    except Exception as error:
        print(f"Internal server error: {error}")
        return jsonify({"error": "เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์"}), 500


# ==================== PromptPay: Generate QR ====================
@app.route('/api/generate-qrcode/<int:order_id>', methods=['GET'])
def api_generate_qrcode(order_id):
    """
    API สำหรับสร้าง PromptPay QR Code สำหรับคำสั่งซื้อ.
    จะสร้าง QR Code ได้หากคำสั่งซื้อเป็นโฆษณาใหม่ที่ 'approved'
    หรือเป็นคำสั่งซื้อต่ออายุที่ 'pending'.
    """
    result = generate_promptpay_qr_for_order(order_id)
    if not result['success']:
        return jsonify(result), 400
    return jsonify({
        'success': True,
        'order_id': order_id,
        'qrcode_base64': result.get('qrcode_base64'),
        'promptpay_payload': result.get('payload')
    })


# ==================== PromptPay: Verify Slip ====================
@app.route('/api/verify-slip/<int:order_id>', methods=['POST'])
def api_verify_slip(order_id):
    """Receive slip + payload, call SlipOK, update order/ad, return short EN messages."""
    file = request.files.get('slip_image')
    if not file or file.filename == '':
        return jsonify({'success': False, 'message': 'No slip received. Attach a slip image.'}), 400

    # เช็กนามสกุลภาพแบบเบาๆ ไม่ให้ไปไกลแล้วค่อยล้ม
    if not _ext_ok(file.filename, ALLOWED_IMAGE_EXTS):
        return jsonify({'success': False, 'message': 'Unsupported slip file type.'}), 400

    payload = request.form.get('payload')
    if not payload:
        return jsonify({'success': False, 'message': 'Missing payload (QR data).'}), 400

    order = find_order_by_id(order_id)
    if not order:
        return jsonify({'success': False, 'message': 'Order not found.'}), 404
    if not can_upload_slip(order):
        return jsonify({'success': False, 'message': 'Slip upload not allowed for this order status.'}), 400

    slip_dir = 'Slip'
    os.makedirs(slip_dir, exist_ok=True)
    unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    save_path = os.path.join(slip_dir, unique_filename)
    file.save(save_path)

    print(f"[INFO] verify-slip: order={order_id}, file={unique_filename}, payload_len={len(payload)}")

    result = verify_payment_and_update_status(order_id, save_path, payload, db)
    status = 200 if result.get('success') else 400
    return jsonify(result), status


# ==================== Internal: Manual run expiry check (for testing) ====================
@app.route('/internal/run-expiry-check', methods=['POST'])
def internal_run_expiry_check():
    """
    Body (JSON) optional: {"now": "YYYY-MM-DD"}  -> ใช้สำหรับ mock วันที่ทดสอบ
    Returns: {count: n, ad_ids: [...]}
    """
    payload = request.get_json(silent=True) or {}
    now_str = payload.get('now')
    try:
        today_date = datetime.fromisoformat(now_str).date() if now_str else date.today()
    except Exception:
        return jsonify({'success': False, 'message': 'invalid date format for now. use YYYY-MM-DD'}), 400

    count, ad_ids = check_ads_expiring_soon(db, today_date)
    return jsonify({'success': True, 'count': count, 'ad_ids': ad_ids}), 200


# ==================== Scheduler bootstrap (ตามเดิม) ====================
_scheduler_started = False

def start_scheduler_once():
    global _scheduler_started
    if not _scheduler_started:
        # ของเดิมมึงใช้ start_expiry_scheduler(app, db) เพราะ job ต้องมี app_context
        start_expiry_scheduler(app, db)
        _scheduler_started = True

# ===== Terminal access log filter (hide only /ai/seen) =====
import logging
import os

def _install_access_log_filter():
    """
    ซ่อนเฉพาะบรรทัด access log ของ /ai/seen ใน terminal
    - คง startup logs (Running on ...) และ access logs endpoint อื่น ๆ
    - ไม่แตะ app.logger, ไม่แตะ request_handler
    """
    class _PathSuppressFilter(logging.Filter):
        def __init__(self, paths):
            super().__init__()
            self.paths = tuple(paths)

        def filter(self, record: logging.LogRecord) -> bool:
            # Access log ของ werkzeug เป็น INFO และมี "HTTP/" อยู่ในข้อความ
            try:
                msg = record.getMessage()
            except Exception:
                msg = str(record.msg)
            if "HTTP/" in msg and any(p in msg for p in self.paths):
                return False
            return True

    # อ่าน path ที่อยากซ่อนจาก env (เผื่ออนาคตอยากเพิ่มหลายอัน)
    suppressed = os.getenv("SUPPRESS_ACCESS_PATHS", "/ai/seen")
    suppressed_paths = [p.strip() for p in suppressed.split(",") if p.strip()]

    wz = logging.getLogger("werkzeug")
    wz.setLevel(logging.INFO)  # คงระดับเดิม เพื่อให้เห็น Running on ...
    f = _PathSuppressFilter(suppressed_paths)

    # ใส่ filter ให้ทุก handler ของ werkzeug
    for h in wz.handlers:
        h.addFilter(f)

    # บางสภาพแวดล้อม werkzeug เขียนผ่าน root handler ด้วย -> ใส่ filter เผื่อ
    logging.getLogger().addFilter(f)

# ===== Terminal access control for /ai/seen =====
import threading, time
from datetime import datetime
from werkzeug.serving import WSGIRequestHandler

# ตั้งค่าด้วย env:
# SUPPRESS_SEEN_ACCESS=1      -> ซ่อน /ai/seen ทุกบรรทัด (default)
# SEEN_ACCESS_SUMMARY_ON_RECOMMEND=1 -> ให้สรุปจำนวน /ai/seen ที่ถูกซ่อน หลัง /ai/recommend ที่ refresh=true
SUPPRESS_SEEN_ACCESS = str(os.getenv("SUPPRESS_SEEN_ACCESS", "1")).lower() in ("1","true","yes","on")
SEEN_ACCESS_SUMMARY_ON_RECOMMEND = str(os.getenv("SEEN_ACCESS_SUMMARY_ON_RECOMMEND", "0")).lower() in ("1","true","yes","on")

_seen_suppressed_counter = 0
_seen_lock = threading.Lock()

def _seen_inc():
    global _seen_suppressed_counter
    with _seen_lock:
        _seen_suppressed_counter += 1

def _seen_pop_count() -> int:
    global _seen_suppressed_counter
    with _seen_lock:
        c = _seen_suppressed_counter
        _seen_suppressed_counter = 0
        return c

class SelectiveWSGIRequestHandler(WSGIRequestHandler):
    """
    ตัด access log เฉพาะ /ai/seen ออกจาก terminal
    (อย่างอื่นแสดงเหมือนเดิม)
    """
    def log(self, type, message, *args):
        try:
            msg = message % args if args else message
        except Exception:
            msg = str(message)
        if "/ai/seen" in msg:
            if SUPPRESS_SEEN_ACCESS:
                if SEEN_ACCESS_SUMMARY_ON_RECOMMEND:
                    _seen_inc()  # เก็บสถิติไว้สรุปตอน refeed
                return  # ไม่พิมพ์บรรทัดนี้ลง terminal
        super().log(type, message, *args)



if __name__ == '__main__':
    # ไม่ต้องไปลดระดับ logger ของ werkzeug อีกแล้ว จะได้เห็น Running on ...
    start_expiry_scheduler(app, db)
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORTAI", "5005")),
        debug=False,
        request_handler=SelectiveWSGIRequestHandler  # << ใช้งาน handler กรอง /ai/seen
    )

