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



import os
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


import os, time, math, json, pickle, random, hashlib, threading
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as sk_normalize
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz, save_npz
from surprise import SVD, Dataset, Reader


# optional deps / locale
try:
    from promptpay import qrcode as promptpay_qrcode
except ImportError:
    promptpay_qrcode = None

try:
    locale.setlocale(locale.LC_ALL, 'th_TH.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'thai')
    except locale.Error:
        print("⚠️ [WARN] Could not set Thai locale. Date formatting might not show full Thai month names.")

app = Flask(__name__)

# ==================== NSFW DETECTION SETUP ====================

# กันรูปยักษ์/ไฟล์ขาดไม่ให้ทำโปรเซสเด้ง (ไม่กระทบผลลัพธ์เดิม)
Image.MAX_IMAGE_PIXELS = 25_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True

# === Config: upload path & model path ===
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = './NSFW_Model/results/20250816-001345/efficientnetb0/models/efficientnetb0_best.pth'

# === Threshold & device (ให้ค่าเดียวกันทั้ง print และ return) ===
HENTAI_THRESHOLD = 34.0
PORN_THRESHOLD = 35.0

# === Model: Custom EfficientNetB0 (โครงเดียวกับตอนเทรน) ===
class CustomEfficientNetB0(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomEfficientNetB0, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=False)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)

# === Labels: mapping id <-> label ===
LABELS = ['normal', 'hentai', 'porn', 'sexy', 'anime']
label2idx = {label: idx for idx, label in enumerate(LABELS)}
idx2label = {idx: label for label, idx in label2idx.items()}

# === Load model: พยายามโหลดอย่างอ่อนโยน ไม่ทำให้โปรเซสตาย ===
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
    # ไม่ exit; ให้ระบบยังรันต่อได้ (จะคืนค่าความน่าจะเป็นเป็นศูนย์ทุกคลาส)
    print(f"โหลดโมเดลไม่สำเร็จ แต่จะไม่ปิดเซิร์ฟเวอร์: {e}")
    MODEL_READY = False

model.eval()

# === Preprocess: ให้ตรงกับตอนเทรน + แก้ EXIF orientation ===
processor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# === Inference API (file path): ตรวจ NSFW ของภาพจากพาธไฟล์ ===
def nude_predict_image(image_path):
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

        hentai_score = probs[label2idx['hentai']] * 100
        porn_score = probs[label2idx['porn']] * 100

        # แสดงผลใน log/console ให้ dev ตามดู (เกณฑ์เดียวกับค่าส่งกลับ)
        is_nsfw = (hentai_score >= HENTAI_THRESHOLD) or (porn_score >= PORN_THRESHOLD)
        print(f"NSFW Detection for {image_path}:")
        print(f"   Hentai: {hentai_score:.2f}%")
        print(f"   Pornography: {porn_score:.2f}%")
        print(f"   Is NSFW: {is_nsfw} (thresholds: hentai>={HENTAI_THRESHOLD} | porn>={PORN_THRESHOLD})")

        # รวมคะแนนทุกคลาสเป็นเปอร์เซ็นต์
        result_dict = {}
        for i in range(len(probs)):
            if i in idx2label:
                result_dict[idx2label[i]] = round(probs[i]*100, 2)
            else:
                result_dict[f"Class_{i}"] = round(probs[i]*100, 2)

        # แปะสถานะ degraded ถ้าโมเดลโหลดไม่ขึ้น (ไม่กระทบฝั่งที่ไม่ใช้คีย์นี้)
        if not MODEL_READY:
            result_dict["__degraded__"] = 1.0

        return is_nsfw, result_dict

    except Exception as e:
        # จับทุกอย่างแล้วคืน error ใน payload แต่ไม่ทำให้เซิร์ฟเวอร์ตาย
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

# ===== Global config =====
DB_URI = 'mysql+mysqlconnector://root:1234@localhost/bestpick'

# ====== Global caches & params (คงชื่อเดิม) ======
recommendation_cache = {}
impression_history_cache = {}

CACHE_EXPIRY_TIME_SECONDS = 120
IMPRESSION_HISTORY_TTL_SECONDS = 3600
IMPRESSION_HISTORY_MAX_ENTRIES = 100

# ====== Lock กัน race condition ======
_cache_lock = threading.Lock()

# ====== Background cache janitor (no log) ======
def clear_cache():
    while True:
        try:
            now = datetime.now()
            with _cache_lock:
                recommendation_cache.clear()
                # TTL prune ของ impression history
                to_delete = []
                for user_id, items in impression_history_cache.items():
                    pruned = [e for e in items if (now - e['timestamp']).total_seconds() < IMPRESSION_HISTORY_TTL_SECONDS]
                    if pruned:
                        impression_history_cache[user_id] = pruned[-IMPRESSION_HISTORY_MAX_ENTRIES:]
                    else:
                        to_delete.append(user_id)
                for uid in to_delete:
                    del impression_history_cache[uid]
        except Exception:
            pass
        finally:
            time.sleep(CACHE_EXPIRY_TIME_SECONDS)

threading.Thread(target=clear_cache, daemon=True).start()

# ====== JWT verify (คง signature เดิม) ======
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


# ===================== RUNTIME RECOMMENDER (no evaluation) =====================


# ====== ใช้ค่าเดิมจาก Global config ที่มึงมี ======
# DB_URI มาจากบล็อก global config ของมึง
CONTENT_VIEW = 'contentbasedview'
EVENT_TABLE  = 'user_interactions'

# คอลัมน์ข้อมูลคอนเทนต์
TEXT_COL = 'Content'
ENGAGE_COL = 'PostEngagement'

# ตัวกรอง action
POS_ACTIONS = {'like','comment','bookmark','share'}
NEG_ACTIONS = {'unlike'}
IGNORE_ACTIONS = {'view_profile','follow','unfollow'}
VIEW_POS_MIN = 3

# น้ำหนักสร้าง implicit rating สำหรับคอลลาบอราทีฟ
ACTION_WEIGHT = {'view':1.0,'like':4.0,'comment':4.0,'bookmark':4.5,'share':5.0,'unlike':-3.0}
RATING_MIN, RATING_MAX = 0.5, 5.0

# พารามิเตอร์ที่ “ล็อกแล้ว” จากการจูน
TFIDF_PARAMS = dict(analyzer='char_wb', ngram_range=(2,5), max_features=60000, min_df=2, max_df=0.95, stop_words=None)
WEIGHTS = (0.3, 0.3, 0.4)  # (collab, item_content, user_content)
POP_ALPHA = 5.0            # Bayesian smoothing สำหรับ engagement

# ไดเรกทอรีแคชไฟล์โมเดล (แชร์กับที่เคยสร้างได้ ไม่งั้นมันจะสร้างใหม่เอง)
OUT_DIR = './recsys_eval_final'
CACHE_DIR = os.path.join(OUT_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# ====== DB ======
def _connect():
    return create_engine(DB_URI, pool_pre_ping=True, pool_recycle=1800)

def _guess_ts_column(eng) -> Optional[str]:
    try:
        one = pd.read_sql(f"SELECT * FROM {EVENT_TABLE} LIMIT 1", eng)
    except Exception:
        return None
    for c in ['created_at','updated_at','ts','timestamp','event_time','inserted_at']:
        if c in one.columns:
            return c
    return None

def _load_content(eng) -> pd.DataFrame:
    df = pd.read_sql(f"SELECT * FROM {CONTENT_VIEW}", eng)
    if 'post_id' not in df.columns and 'id' in df.columns:
        df = df.rename(columns={'id':'post_id'})
    df['post_id'] = pd.to_numeric(df['post_id'], errors='coerce').dropna().astype(int)

    if TEXT_COL not in df.columns: df[TEXT_COL] = ''
    if ENGAGE_COL not in df.columns: df[ENGAGE_COL] = 0.0
    eng = pd.to_numeric(df[ENGAGE_COL], errors='coerce').fillna(0.0).astype(np.float32)

    # Popularity prior + normalized engagement
    prior = (eng + POP_ALPHA) / (eng.max() + POP_ALPHA)
    df['PopularityPrior'] = _normalize(pd.Series(prior))
    df['NormalizedEngagement'] = _normalize(pd.Series(eng))

    if 'owner_id' not in df.columns:
        df['owner_id'] = -1
    else:
        df['owner_id'] = pd.to_numeric(df['owner_id'], errors='coerce').fillna(-1).astype(int)

    return df

def _load_events(eng, ts_col: Optional[str]) -> pd.DataFrame:
    base = "user_id, post_id, action_type"
    if ts_col: base += f", {ts_col} AS ts"
    ev = pd.read_sql(f"SELECT {base} FROM {EVENT_TABLE}", eng)
    ev = ev.dropna(subset=['user_id','post_id'])
    ev['user_id'] = pd.to_numeric(ev['user_id'], errors='coerce').dropna().astype(int)
    ev['post_id'] = pd.to_numeric(ev['post_id'], errors='coerce').dropna().astype(int)
    ev['action_type'] = ev['action_type'].astype(str).str.lower()
    if 'ts' in ev.columns: ev = ev.dropna(subset=['ts'])
    ev = ev[~ev['action_type'].isin(IGNORE_ACTIONS)].copy()
    return ev

# ====== Utils ======
def _normalize(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce').fillna(0.0).astype(np.float32)
    mn, mx = float(s.min()), float(s.max())
    return (s - mn) / (mx - mn + 1e-12)

def _md5_of_df(df: pd.DataFrame, cols: List[str]) -> str:
    snap = df[cols].copy().fillna(0)
    h = hashlib.md5(pd.util.hash_pandas_object(snap, index=False).values).hexdigest()
    return h

# ====== Content models ======
def _build_tfidf(content_df: pd.DataFrame, params: dict):
    tfidf = TfidfVectorizer(**params, dtype=np.float32)
    X = tfidf.fit_transform(content_df[TEXT_COL].fillna(''))
    return tfidf, X.astype(np.float32)

def _build_knn(X):
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn.fit(X)
    return knn

def _precompute_item_content_scores(knn, content_df: pd.DataFrame, X: csr_matrix) -> np.ndarray:
    n = X.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    n_neighbors = min(20, n)
    dists, idxs = knn.kneighbors(X, n_neighbors=n_neighbors)
    eng = content_df['NormalizedEngagement'].to_numpy(dtype=np.float32)
    for i in range(n):
        scores[i] = float(np.mean(eng[idxs[i]])) if idxs[i].size else 0.0
    return scores

def _user_text_profiles(train_pos: pd.DataFrame, content_df: pd.DataFrame, X: csr_matrix) -> Dict[int, csr_matrix]:
    pid_to_idx = {int(pid): i for i, pid in enumerate(content_df['post_id'].tolist())}
    profiles = {}
    for uid, g in train_pos.groupby('user_id'):
        idxs = [pid_to_idx.get(p) for p in g['post_id'].tolist() if pid_to_idx.get(p) is not None]
        if not idxs:
            profiles[int(uid)] = csr_matrix((1, X.shape[1]), dtype=np.float32)
            continue
        mat = X[idxs]
        mean_vec = mat.mean(axis=0)
        if hasattr(mean_vec, "toarray"): mean_vec = mean_vec.toarray()
        else: mean_vec = np.asarray(mean_vec)
        prof = sk_normalize(mean_vec)
        profiles[int(uid)] = csr_matrix(prof, dtype=np.float32)
    return profiles

def _user_content_score(uid: int, profiles: Dict[int, csr_matrix], X: csr_matrix, idx: int) -> float:
    prof = profiles.get(int(uid))
    if prof is None or prof.nnz == 0: return 0.0
    v = X[idx]
    num = float(v.multiply(prof).sum())
    den = (np.linalg.norm(v.data) * np.linalg.norm(prof.data)) if prof.nnz>0 and v.nnz>0 else 0.0
    return float(num/den) if den>0 else 0.0

# ====== Collab ======
def _build_true_labels(events: pd.DataFrame) -> pd.DataFrame:
    t = events.groupby(['user_id','post_id','action_type']).size().reset_index(name='cnt')
    if t.empty:
        return pd.DataFrame(columns=['user_id','post_id','y'])
    pvt = t.pivot_table(index=['user_id','post_id'], columns='action_type',
                        values='cnt', fill_value=0, aggfunc='sum').reset_index()
    pvt.columns = [str(c).lower() for c in pvt.columns]
    pos = np.zeros(len(pvt), dtype=bool)
    for a in POS_ACTIONS:
        if a in pvt.columns: pos |= (pvt[a].to_numpy(dtype=float) > 0)
    if 'view' in pvt.columns:
        pos |= (pvt['view'].to_numpy(dtype=float) >= VIEW_POS_MIN)
    if NEG_ACTIONS:
        neg = np.zeros(len(pvt), dtype=bool)
        for a in NEG_ACTIONS:
            if a in pvt.columns: neg |= (pvt[a].to_numpy(dtype=float) > 0)
        pos = np.where(neg, False, pos)
    pvt['y'] = pos.astype(int)
    return pvt[['user_id','post_id','y']]

def _build_collab_model(events: pd.DataFrame, post_ids: List[int]):
    e = events[events['post_id'].isin(post_ids)].copy()
    if e.empty: return None
    t = e.groupby(['user_id','post_id','action_type']).size().reset_index(name='cnt')
    pvt = t.pivot_table(index=['user_id','post_id'], columns='action_type',
                        values='cnt', fill_value=0, aggfunc='sum').reset_index()

    rating = np.zeros(len(pvt), dtype=np.float32)
    for act, w in ACTION_WEIGHT.items():
        if act in pvt.columns:
            rating += np.float32(w) * pvt[act].to_numpy(dtype=np.float32)
    if 'view' in pvt.columns:
        rating += np.where(pvt['view'].to_numpy(dtype=np.float32) >= VIEW_POS_MIN, np.float32(2.0), np.float32(0.0))
    rating = np.clip(rating, RATING_MIN, RATING_MAX)

    data = pvt[['user_id','post_id']].copy()
    data['rating'] = rating
    data = data[data['rating'] > 0]
    if data.empty: return None

    reader = Reader(rating_scale=(RATING_MIN, RATING_MAX))
    dset = Dataset.load_from_df(data[['user_id','post_id','rating']], reader)
    trainset = dset.build_full_trainset()
    model = SVD(n_factors=150, n_epochs=60, lr_all=0.005, reg_all=0.5)
    model.fit(trainset)
    return model

# ====== Ranking ======
def _recommend_scores_for_user(uid: int,
                               content_df: pd.DataFrame,
                               tfidf, X, knn,
                               uc_profiles: Dict[int, csr_matrix],
                               collab_model,
                               item_content_scores: np.ndarray,
                               weights: Tuple[float,float,float]) -> pd.DataFrame:
    wc, wi, wu = weights
    rows = []
    collab_pred_default = 0.5
    for i in range(len(content_df)):
        row = content_df.iloc[i]
        pid = int(row['post_id'])
        # กันโพสต์ของตัวเองออกจากแคนดิเดต
        if int(row.get('owner_id', -1)) == int(uid):
            continue
        # collab
        collab = collab_pred_default
        if collab_model is not None:
            try:
                collab = float(collab_model.predict(int(uid), pid).est)
            except Exception:
                collab = collab_pred_default
        # item content
        ic = float(item_content_scores[i]) if i < len(item_content_scores) else 0.0
        # user-content cosine
        uc = _user_content_score(uid, uc_profiles, X, i)
        # popularity prior เสริม
        pop = float(row.get('PopularityPrior', 0.0))
        final = wc*collab + wi*ic + wu*uc + 0.05*pop
        rows.append((pid, collab, ic, uc, pop, final))
    out = pd.DataFrame(rows, columns=['post_id','collab','item_content','user_content','pop','final'])
    out['final_norm'] = _normalize(out['final'])
    return out.sort_values(['final_norm','final'], ascending=[False, False])

# ====== Model block cache (ไฟล์) ======
def _get_cache_paths(cache_key: str, tfidf_params: dict):
    key = hashlib.md5(json.dumps(tfidf_params, sort_keys=True).encode()).hexdigest()
    base = os.path.join(CACHE_DIR, cache_key)
    return {
        'tfidf_pkl': base + f'.tfidf_{key}.pkl',
        'X_npz'    : base + f'.X_{key}.npz',
        'knn_pkl'  : base + '.knn.pkl',
        'ics_npy'  : base + '.item_scores.npy',
        'uc_pkl'   : base + '.ucprof.pkl',
        'svd_pkl'  : base + '.svd.pkl',
    }

def _build_or_load_blocks(content_df: pd.DataFrame, events: pd.DataFrame, use_cache: bool=True):
    content_hash = _md5_of_df(content_df, ['post_id', TEXT_COL, ENGAGE_COL])
    sample_ev = events[['user_id','post_id','action_type']].head(5000).copy() if len(events)>5000 else events[['user_id','post_id','action_type']]
    events_hash  = _md5_of_df(sample_ev, ['user_id','post_id','action_type'])
    cache_key = f"{content_hash}_{events_hash}"
    P = _get_cache_paths(cache_key, TFIDF_PARAMS)

    # TF-IDF
    if use_cache and os.path.exists(P['tfidf_pkl']) and os.path.exists(P['X_npz']):
        with open(P['tfidf_pkl'],'rb') as f: tfidf = pickle.load(f)
        X = load_npz(P['X_npz']).astype(np.float32)
    else:
        tfidf, X = _build_tfidf(content_df, TFIDF_PARAMS)
        with open(P['tfidf_pkl'],'wb') as f: pickle.dump(tfidf,f)
        save_npz(P['X_npz'], X)

    # KNN + item content
    if use_cache and os.path.exists(P['knn_pkl']) and os.path.exists(P['ics_npy']):
        with open(P['knn_pkl'],'rb') as f: knn = pickle.load(f)
        item_scores = np.load(P['ics_npy'])
    else:
        knn = _build_knn(X)
        item_scores = _precompute_item_content_scores(knn, content_df, X)
        with open(P['knn_pkl'],'wb') as f: pickle.dump(knn,f)
        np.save(P['ics_npy'], item_scores)

    # user text profiles (จาก label บวก)
    labels = _build_true_labels(events)
    train_pos = labels[labels['y']==1][['user_id','post_id']]
    if use_cache and os.path.exists(P['uc_pkl']):
        try:
            with open(P['uc_pkl'],'rb') as f: uc_prof = pickle.load(f)
        except Exception:
            uc_prof = _user_text_profiles(train_pos, content_df, X)
            with open(P['uc_pkl'],'wb') as f: pickle.dump(uc_prof,f)
    else:
        uc_prof = _user_text_profiles(train_pos, content_df, X)
        with open(P['uc_pkl'],'wb') as f: pickle.dump(uc_prof,f)

    # collab
    if use_cache and os.path.exists(P['svd_pkl']):
        try:
            with open(P['svd_pkl'],'rb') as f: svd = pickle.load(f)
        except Exception:
            svd = _build_collab_model(events, content_df['post_id'].tolist())
            with open(P['svd_pkl'],'wb') as f: pickle.dump(svd,f)
    else:
        svd = _build_collab_model(events, content_df['post_id'].tolist())
        with open(P['svd_pkl'],'wb') as f: pickle.dump(svd,f)

    return tfidf, X, knn, item_scores, uc_prof, svd, cache_key

# ====== Public runtime API (เรียกใช้ภายในแอป) ======
def recommend_for_user(user_id: int, top_k: int = 20, use_cache: bool=True) -> List[int]:
    """
    คืนรายการ post_id ที่จัดอันดับแล้ว พร้อมใช้กับ split_and_rank_recommendations
    อาศัยแคช in-memory (ของมึง) + แคชไฟล์บล็อกโมเดล เพื่อลด latency
    """
    now = datetime.now()

    # in-memory cache hit (ล้างโดยเจ้า background janitor ของมึงอยู่แล้ว)
    with _cache_lock:
        cached = recommendation_cache.get(user_id)
        if cached:
            # ตัดตาม top_k ฝั่ง client
            ids = cached['ids'][:top_k]
            return [int(x) for x in ids]

    # เตรียมดาต้า
    eng = _connect()
    content_df = _load_content(eng)
    ts_col = _guess_ts_column(eng)
    events = _load_events(eng, ts_col)

    # สร้าง/โหลดบล็อกโมเดล
    tfidf, X, knn, item_scores, uc_prof, svd, _ = _build_or_load_blocks(content_df, events, use_cache=use_cache)

    # สกอร์พื้นฐานเรียงจากสูงไปต่ำ
    sc = _recommend_scores_for_user(int(user_id), content_df, tfidf, X, knn, uc_prof, svd, item_scores, WEIGHTS)

    # รายการที่ผู้ใช้เคยมีปฏิสัมพันธ์แล้ว (กันลูปเดิม)
    user_interacted = events[events['user_id']==int(user_id)]['post_id'].tolist()

    # impression history (TTL/trim จัดการโดย thread ของมึงแล้ว)
    with _cache_lock:
        impression_history = impression_history_cache.get(int(user_id), []).copy()

    total_posts = int(content_df['post_id'].nunique())

    # เอา candidate ทั้งหมดไปจัดลำดับใหม่ตาม split & rank
    ranked_all = split_and_rank_recommendations(
        recommendations=sc['post_id'].tolist(),
        user_interactions=user_interacted,
        impression_history=impression_history,
        total_posts_in_db=total_posts
    )

    # หั่นตาม top_k
    final_ids = [int(p) for p in ranked_all[:max(top_k, 1)]]

    # อัพเดต in-memory cache
    with _cache_lock:
        recommendation_cache[user_id] = {'ids': ranked_all, 'timestamp': now}

        # อัพเดต impression history ด้วยโพสต์ที่เพิ่งส่งไป
        hist = impression_history_cache.get(user_id, [])
        now_ts = datetime.now()
        for pid in final_ids:
            hist.append({'post_id': int(pid), 'timestamp': now_ts})
        # เก็บล่าสุดไม่เกิน IMPRESSION_HISTORY_MAX_ENTRIES (ตัว clean-up จะ prune ตาม TTL อยู่แล้ว)
        impression_history_cache[user_id] = hist[-IMPRESSION_HISTORY_MAX_ENTRIES:]

    return final_ids


# ==================== Split & Rank (ของเดิม, ตัด log ออก) ====================
def split_and_rank_recommendations(recommendations, user_interactions, impression_history, total_posts_in_db):
    unique_recommendations_ids = [int(p) if isinstance(p, float) else p for p in list(dict.fromkeys(recommendations))]
    user_interactions_set = set([int(p) if isinstance(p, float) else p for p in user_interactions])
    impression_history_set = set([int(e['post_id']) for e in impression_history])

    final_recommendations_ordered = []

    truly_unviewed_posts = [
        post_id for post_id in unique_recommendations_ids
        if post_id not in user_interactions_set and post_id not in impression_history_set
    ]

    recently_shown_not_interacted = [
        post_id for post_id in unique_recommendations_ids
        if post_id not in user_interactions_set and post_id in impression_history_set
    ]

    interacted_posts = [
        post_id for post_id in unique_recommendations_ids
        if post_id in user_interactions_set
    ]

    num_fresh_priority = min(len(truly_unviewed_posts), max(10, int(len(unique_recommendations_ids) * 0.25)))
    final_recommendations_ordered.extend(truly_unviewed_posts[:num_fresh_priority])

    remaining_posts_to_mix = [
        post_id for post_id in unique_recommendations_ids
        if post_id not in set(final_recommendations_ordered)
    ]

    group_A_not_recently_shown = [p for p in remaining_posts_to_mix if p not in impression_history_set]
    group_B_recently_shown = [p for p in remaining_posts_to_mix if p in impression_history_set]

    num_to_demote_from_history = min(len(impression_history), int(len(impression_history) * 0.25))
    posts_to_demote = set([entry['post_id'] for entry in impression_history[:num_to_demote_from_history]])

    demoted_posts, non_demoted_posts = [], []
    for post_id in unique_recommendations_ids:
        (demoted_posts if post_id in posts_to_demote else non_demoted_posts).append(post_id)

    truly_unviewed_non_demoted = [p for p in non_demoted_posts if p not in user_interactions_set and p not in impression_history_set]
    remaining_non_demoted_and_not_truly_unviewed = [p for p in non_demoted_posts if p not in set(truly_unviewed_non_demoted)]

    num_unviewed_first_current_run = min(30, len(truly_unviewed_non_demoted))
    final_recommendations_ordered.extend(truly_unviewed_non_demoted[:num_unviewed_first_current_run])

    remaining_to_shuffle_and_mix = (
        truly_unviewed_non_demoted[num_unviewed_first_current_run:]
        + remaining_non_demoted_and_not_truly_unviewed
        + demoted_posts
    )

    shuffled_segment = []
    block_size = 5
    blocks = [remaining_to_shuffle_and_mix[i:i + block_size] for i in range(0, len(remaining_to_shuffle_and_mix), block_size)]

    for block in blocks:
        block_with_priority = []
        for post_id in block:
            priority_score = 2 if post_id in posts_to_demote else (1 if post_id in impression_history_set else 0)
            block_with_priority.append((post_id, priority_score))
        block_with_priority.sort(key=lambda x: (x[1], random.random()))
        shuffled_segment.extend([post_id for post_id, _ in block_with_priority])

    final_recommendations_ordered.extend(shuffled_segment)
    return final_recommendations_ordered



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

# ==================== Recommendation Route (optimized, production) ====================

@app.route('/ai/recommend', methods=['POST'])
@verify_token
def recommend():
    try:
        user_id = int(request.user_id)
        if user_id <= 0:
            return jsonify({"error": "Invalid user"}), 400

        # คุมจำนวนโพสต์ที่ดึงกลับ
        top_k = int(request.args.get('top_k', 20))
        top_k = max(1, min(100, top_k))

        # refresh=true จะบังคับไม่ใช้แคชในรอบนี้ และเคลียร์ in-memory ids cache ของ user
        refresh_requested = request.args.get('refresh', 'false').lower() == 'true'
        if refresh_requested:
            with _cache_lock:
                recommendation_cache.pop(user_id, None)  # เคลียร์ ids cache ของระบบใหม่เท่านั้น

        # เรียก runtime core ที่จัดการหมดแล้ว (rank + splitrank + impression history)
        ids = recommend_for_user(user_id=user_id, top_k=top_k, use_cache=(not refresh_requested))
        if not ids:
            return jsonify([]), 200

        # ดึงข้อมูลโพสต์ตามลำดับที่ AI จัดมา
        placeholders = ', '.join([f':id_{i}' for i in range(len(ids))])
        with db.session.begin():
            q = text(f"""
                SELECT p.*, u.username, u.picture,
                       (SELECT COUNT(*) FROM likes WHERE post_id = p.id AND user_id = :user_id) AS is_liked
                FROM posts p
                JOIN users u ON p.user_id = u.id
                WHERE p.status = 'active' AND p.id IN ({placeholders})
            """)
            params = {'user_id': user_id, **{f'id_{i}': pid for i, pid in enumerate(ids)}}
            rows = db.session.execute(q, params).fetchall()

        # คืนลำดับให้ตรงกับ ids
        order_map = {pid: i for i, pid in enumerate(ids)}
        posts = [r._mapping for r in rows]
        posts_sorted = sorted(posts, key=lambda x: order_map.get(x['id'], 10**9))

        # สร้าง payload
        out = []
        for post in posts_sorted:
            try:
                updated = post.get('updated_at')
                if updated is None:
                    iso_updated = datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'
                else:
                    if getattr(updated, 'tzinfo', None) is None:
                        updated = updated.replace(tzinfo=timezone.utc)
                    iso_updated = updated.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
            except Exception:
                iso_updated = datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'

            out.append({
                "id": post['id'],
                "userId": post['user_id'],
                "title": post.get('Title'),
                "content": post.get('content'),
                "updated": iso_updated,
                "photo_url": json.loads(post.get('photo_url', '[]') or '[]'),
                "video_url": json.loads(post.get('video_url', '[]') or '[]'),
                "userName": post['username'],
                "userProfileUrl": post['picture'],
                "is_liked": (post['is_liked'] or 0) > 0
            })

        return jsonify(out), 200

    except Exception as e:
        # อยาก log ก็ใส่ logger.error(str(e)) เอาเอง
        return jsonify({"error": "Internal Server Error"}), 500


# === NSFW Detection Route: create post ===
@app.route('/ai/posts/create', methods=['POST'])
def create_post():
    try:
        user_id = request.form.get('user_id')
        content = request.form.get('content')
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
            if not photo or not photo.filename:
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
            except Exception as e:
                print(f"Error processing photo {getattr(photo, 'filename', '?')}: {e}")
                if photo_path and os.path.exists(photo_path):
                    os.remove(photo_path)
                invalid_photos.append({
                    "filename": getattr(photo, 'filename', '?'),
                    "reason": "Unable to process the image.",
                    "details": {"error": str(e)}
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
            if not video or not video.filename:
                continue
            try:
                vname, vpath = _save_upload(video, ALLOWED_VIDEO_EXTS, UPLOAD_FOLDER)
                video_urls.append(f'/uploads/{vname}')
            except Exception as e:
                print(f"Skip invalid video {getattr(video, 'filename', '?')}: {e}")
                # ไม่ fail ทั้งโพสต์เพราะวิดีโอพัง

        photo_urls_json = json.dumps(photo_urls)
        video_urls_json = json.dumps(video_urls)

        try:
            insert_query = text("""
                INSERT INTO posts (user_id, Title, content, ProductName, CategoryID, photo_url, video_url, status, updated_at)
                VALUES (:user_id, :title, :content, :product_name, :category_id, :photo_urls, :video_urls, 'active', NOW())
            """)
            result = db.session.execute(insert_query, {
                'user_id': user_id,
                'title': title,
                'content': content,
                'product_name': product_name,
                'category_id': category,
                'photo_urls': photo_urls_json,
                'video_urls': video_urls_json
            })
            db.session.commit()

            post_id = getattr(result, "lastrowid", None)
            if not post_id:
                # fallback สำหรับบาง dialect
                post_id = db.session.execute(text("SELECT LAST_INSERT_ID()")).scalar()

            print(f"Post created successfully with ID: {post_id}, {len(photo_urls)} photos and {len(video_urls)} videos")

            return jsonify({
                "post_id": post_id,
                "user_id": user_id,
                "content": content,
                "category": category,
                "Title": title,
                "ProductName": product_name,
                "video_urls": video_urls,
                "photo_urls": photo_urls
            }), 201

        except Exception as db_error:
            print(f"Database error: {db_error}")
            db.session.rollback()
            # ลบไฟล์ที่เพิ่งเซฟเพื่อลดขยะในกรณี DB fail
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

    except Exception as error:
        print("Internal server error:", str(error))
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

if __name__ == '__main__':
    # คงพฤติกรรมเดิม: เรียกด้วย app, db
    start_expiry_scheduler(app, db)
    app.run(host='0.0.0.0', port=5005)
