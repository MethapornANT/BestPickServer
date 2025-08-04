from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import json
import requests
import time
import threading
import re
import traceback
import joblib
import pandas as pd
import jwt
import random
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import locale # สำหรับการแสดงวันที่ภาษาไทย

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


from datetime import datetime, timezone
from datetime import datetime, timedelta, date, timezone
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from pythainlp.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
from surprise import SVD, Dataset, Reader
from textblob import TextBlob
from pythainlp import word_tokenize
from functools import wraps
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

import qrcode
from qrcode.constants import ERROR_CORRECT_H
import uuid
import base64
import io
try:
    from promptpay import qrcode as promptpay_qrcode
except ImportError:
    promptpay_qrcode = None  # จะ refactor ให้รองรับกรณีไม่มี promptpay ทีหลัง

try:
    locale.setlocale(locale.LC_ALL, 'th_TH.UTF-8') # ลองใช้สำหรับ Linux/macOS
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'thai') # ลองใช้สำหรับ Windows
    except locale.Error:
        print("⚠️ [WARN] Could not set Thai locale. Date formatting might not show full Thai month names.")


app = Flask(__name__)

# ==================== NSFW DETECTION SETUP ====================

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# โหลดโมเดลและ processor สำหรับ EfficientNetB0
# กำหนดพาธของโมเดลให้ถูกต้องตามที่คุณให้มา
MODEL_PATH = './NSFW_Model_archived_20250803-132149/results/20250801-100650 Best Model/efficientnetb0/models/efficientnetb0_best.pth'

# กำหนดโครงสร้าง EfficientNetB0 ที่ใช้ตอนเซฟโมเดล
# สมมติว่าคุณเปลี่ยน classifier layer ให้มี output 5 คลาส
class CustomEfficientNetB0(nn.Module):
    def __init__(self, num_classes=5): # กำหนดจำนวนคลาสตามที่คุณเทรน
        super(CustomEfficientNetB0, self).__init__()
        # efficientnet_b0 จะถูกสร้างขึ้นมาเป็นส่วนหนึ่งของ CustomEfficientNetB0
        self.efficientnet = models.efficientnet_b0(pretrained=False) # ไม่ใช้ pretrained weights จาก torchvision
        # ส่วนนี้ต้องตรงกับที่คุณแก้ในโมเดลตอนเทรนใน NSFW.py
        # คือการเปลี่ยน classifier[1]
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)

# mapping id เป็น label ให้ตรงกับที่คุณใช้ตอนเทรน
LABELS = ['normal', 'hentai', 'porn', 'sexy', 'anime']
label2idx = {label: idx for idx, label in enumerate(LABELS)}
idx2label = {idx: label for label, idx in label2idx.items()}

# โหลดโมเดล EfficientNetB0 ที่เทรนแล้ว
# สร้าง CustomEfficientNetB0 ก่อน
model = CustomEfficientNetB0(num_classes=len(LABELS)) # จำนวนคลาสต้องตรงกับที่คุณเทรน (5 คลาส)

try:
    # **** ส่วนที่แก้ไข: โหลด state_dict เข้าสู่ self.efficientnet โดยตรง ****
    # ไฟล์ .pth ของคุณน่าจะบันทึก state_dict ของ `models.efficientnet_b0` โดยตรง
    # ไม่ใช่ state_dict ของ `CustomEfficientNetB0` ทั้งก้อน
    state_dict_from_file = torch.load(MODEL_PATH)
    model.efficientnet.load_state_dict(state_dict_from_file) # โหลดเข้าสู่ EfficientNet ภายใน CustomEfficientNetB0
    print("โหลด state_dict เข้าสู่ model.efficientnet โดยตรงสำเร็จงับ!")

except Exception as e:
    print(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    print("โปรดตรวจสอบว่าไฟล์โมเดลถูกต้องและโครงสร้างโมเดล 'CustomEfficientNetB0' ตรงกับที่ใช้ในการเทรนงับ")
    # คุณอาจจะต้องเพิ่มการจัดการข้อผิดพลาดที่นี่ หรือทำให้โปรแกรมหยุดทำงาน
    exit() # หยุดการทำงานหากโหลดโมเดลไม่ได้

model.eval() # ตั้งค่าโมเดลเป็นโหมดประเมินผล

# กำหนด transform สำหรับ preprocessing รูปภาพให้เข้ากับ EfficientNetB0
# ต้องตรงกับที่คุณใช้ตอน Train โมเดล EfficientNetB0 ใน NSFW.py
processor = transforms.Compose([
    transforms.Resize((224, 224)), # ขนาดที่ใช้ใน NSFW.py คือ 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ฟังก์ชันนี้ใช้สำหรับตรวจจับภาพที่ไม่เหมาะสม (NSFW)
def nude_predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image).unsqueeze(0) # เพิ่ม batch dimension
        
        # ย้าย input ไปยัง device ที่โมเดลอยู่ (CPU/GPU)
        device = next(model.parameters()).device # ตรวจสอบ device ที่โมเดลกำลังใช้งานอยู่
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            logits = outputs # EfficientNetB0 โดยทั่วไปจะให้ logits ตรงๆ
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

        # ดึงคะแนน Hentai และ Pornography โดยใช้อ้างอิงจาก label2idx
        # ตรวจสอบว่า index มีอยู่ใน list ของ probs หรือไม่
        if label2idx['hentai'] >= len(probs) or label2idx['porn'] >= len(probs):
             raise IndexError("ไม่พบ Index ของ 'hentai' หรือ 'porn' ในผลลัพธ์การคาดการณ์งับ")

        hentai_score = probs[label2idx['hentai']] * 100
        porn_score = probs[label2idx['porn']] * 100

        print(f"NSFW Detection for {image_path}:")
        print(f"   Hentai: {hentai_score:.2f}%")
        print(f"   Pornography: {porn_score:.2f}%")
        print(f"   Is NSFW: {hentai_score > 20 or porn_score > 20}")

        # สร้าง dictionary ของผลลัพธ์
        result_dict = {}
        for i in range(len(probs)):
            if i in idx2label:
                result_dict[idx2label[i]] = round(probs[i]*100, 2)
            else:
                result_dict[f"Class_{i}"] = round(probs[i]*100, 2) # กรณีไม่เจอ label (ไม่น่าจะเกิดขึ้นถ้า LABELS ถูกต้อง)
        
        return hentai_score > 20 or porn_score > 20, result_dict
    except Exception as e:
        print(f"Error in NSFW detection for {image_path}: {e}")
        return False, {"error": f"ไม่สามารถตรวจสอบภาพได้งับ: {e}"}

# ==================== DATABASE SETUP ====================
# กำหนดค่า URI สำหรับการเชื่อมต่อฐานข้อมูล
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:1234@localhost/bestpick'

# เริ่มต้นใช้งาน SQLAlchemy
db = SQLAlchemy(app)

# ==================== SQLAlchemy Models สำหรับ Slip/Order/Ad ====================

# โมเดลสำหรับตาราง Order (คำสั่งซื้อ) ในฐานข้อมูล
# ใช้เก็บข้อมูลเกี่ยวกับการสั่งซื้อแพ็กเกจโฆษณา รวมถึงสถานะการชำระเงินและรายละเอียดอื่นๆ
class Order(db.Model):
    __tablename__ = 'orders'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)

    renew_ads_id = db.Column(db.Integer, nullable=True)
    package_id = db.Column(db.Integer, nullable=True)

    amount = db.Column(db.Numeric(10,2), nullable=False)
    promptpay_qr_payload = db.Column(db.String(255), nullable=True)

    status = db.Column(db.Enum('pending', 'approved', 'paid', 'active', 'rejected', 'expired'), nullable=False, default='pending')

    slip_image = db.Column(db.String(255), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    show_at = db.Column(db.Date, nullable=True)

    def __repr__(self):
        return f'<Order {self.id}>'


# โมเดลสำหรับตาราง Ad (โฆษณา) ในฐานข้อมูล
# ใช้เก็บข้อมูลรายละเอียดของโฆษณาที่ผู้ใช้สร้างขึ้น รวมถึงสถานะการแสดงผลและข้อมูลสำหรับผู้ดูแลระบบ
class Ad(db.Model):
    __tablename__ = 'ads'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    order_id = db.Column(db.Integer, nullable=False)

    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    link = db.Column(db.String(255), nullable=True)
    image = db.Column(db.String(255), nullable=True)

    status = db.Column(db.Enum('pending', 'approved', 'paid', 'active', 'rejected', 'expired'), nullable=False, default='pending')

    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    expiration_date = db.Column(db.Date, nullable=True)

    admin_notes = db.Column(db.Text, nullable=True)
    admin_slip = db.Column(db.String(255), nullable=True)

    show_at = db.Column(db.Date, nullable=True)

    def __repr__(self):
        return f'<Ad {self.id}>'


# โมเดลสำหรับตาราง AdPackage (แพ็กเกจโฆษณา) ในฐานข้อมูล
# ใช้เก็บข้อมูลแพ็กเกจโฆษณาที่มีให้เลือก เช่น ชื่อแพ็กเกจ ราคา และระยะเวลา
class AdPackage(db.Model):
    __tablename__ = 'ad_packages'
    package_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Numeric(10, 2), nullable=False)
    duration_days = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<AdPackage {self.package_id}>'

load_dotenv()
# Secret key สำหรับการเข้ารหัส/ถอดรหัสโทเค็น JWT
JWT_SECRET = os.getenv('JWT_SECRET')


# ==================== RECOMMENDATION SYSTEM FUNCTIONS ====================

# Global cache variables
recommendation_cache = {}
impression_history_cache = {}

# Cache expiry times
CACHE_EXPIRY_TIME_SECONDS = 120   # ระยะเวลาหมดอายุของแคชคำแนะนำหลัก
IMPRESSION_HISTORY_TTL_SECONDS = 3600 # ระยะเวลาเก็บประวัติการแสดงผล (TTL)
IMPRESSION_HISTORY_MAX_ENTRIES = 100 # จำนวนสูงสุดของรายการประวัติการแสดงผลต่อผู้ใช้

# ฟังก์ชันนี้มีไว้สำหรับล้างแคชคำแนะนำและแคชประวัติการแสดงผลแบบวนลูป
# เพื่อให้ข้อมูลแคชเป็นปัจจุบันและป้องกันหน่วยความจำล้น
def clear_cache():
    global recommendation_cache, impression_history_cache
    while True:
        now = datetime.now()

        recommendation_cache = {}

        for user_id in list(impression_history_cache.keys()):
            impression_history_cache[user_id] = [
                entry for entry in impression_history_cache[user_id]
                if (now - entry['timestamp']).total_seconds() < IMPRESSION_HISTORY_TTL_SECONDS
            ]
            if not impression_history_cache[user_id]:
                del impression_history_cache[user_id]

        time.sleep(CACHE_EXPIRY_TIME_SECONDS)

threading.Thread(target=clear_cache, daemon=True).start()

# ฟังก์ชันนี้ใช้สำหรับตรวจสอบความถูกต้องของโทเค็น JWT ใน Header ของ Request
# เพื่อยืนยันตัวตนและสิทธิ์ของผู้ใช้งานก่อนเข้าถึง API
def verify_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "No token provided or incorrect format"}), 403

        token = auth_header.split(" ")[1]
        try:
            decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            request.user_id = decoded.get("id")
            request.role = decoded.get("role")
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Unauthorized: Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Unauthorized: Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated_function

# ฟังก์ชันนี้มีหน้าที่โหลดข้อมูล Content-Based และ Collaborative Filtering จากฐานข้อมูล MySQL
# เพื่อเตรียมข้อมูลสำหรับการสร้างโมเดลและให้คำแนะนำ
def load_data_from_db():
    try:
        engine = create_engine('mysql+mysqlconnector://root:1234@localhost/bestpick')
        query_content = "SELECT * FROM contentbasedview;"
        content_based_data = pd.read_sql(query_content, con=engine)
        print("โหลดข้อมูล Content-Based สำเร็จ")
        query_collaborative = "SELECT * FROM collaborativeview;"
        collaborative_data = pd.read_sql(query_collaborative, con=engine)
        print("โหลดข้อมูล Collaborative สำเร็จ")
        return content_based_data, collaborative_data
    except Exception as e:
        print(f"ข้อผิดพลาดในการโหลดข้อมูลจากฐานข้อมูล: {str(e)}")
        raise

# ฟังก์ชันนี้ใช้สำหรับปรับค่าคะแนนให้เป็นมาตรฐาน (Normalization)
# โดยจะปรับค่าให้อยู่ในช่วง 0 ถึง 1 เพื่อให้สามารถเปรียบเทียบหรือรวมกันได้
def normalize_scores(series):
    min_val, max_val = series.min(), series.max()
    if max_val > min_val:
        return (series - min_val) / (max_val - min_val)
    return series

# ฟังก์ชันนี้ใช้สำหรับปรับค่าการมีส่วนร่วม (Engagement) ให้เป็นมาตรฐานสำหรับแต่ละผู้ใช้งาน
# เพื่อให้คะแนนการมีส่วนร่วมสามารถนำไปใช้ในการคำนวณคะแนนรวมได้อย่างเหมาะสม
def normalize_engagement(data, user_column='owner_id', engagement_column='PostEngagement'):
    data['NormalizedEngagement'] = data.groupby(user_column)[engagement_column].transform(lambda x: normalize_scores(x))
    return data

# ฟังก์ชันนี้ใช้สำหรับนับจำนวนคอมเมนต์จากข้อความคอมเมนต์ที่คั่นด้วยเซมิโคลอน
# เพื่อนำจำนวนคอมเมนต์ไปใช้ในการคำนวณคะแนนการมีส่วนร่วม
def analyze_comments(comments_series):
    comment_counts = []
    for comment_text in comments_series:
        if pd.isna(comment_text) or str(comment_text).strip() == '':
            comment_counts.append(0)
        else:
            individual_comments = [c.strip() for c in str(comment_text).split(';') if c.strip()]
            comment_counts.append(len(individual_comments))
    return comment_counts

# ฟังก์ชันนี้ใช้สำหรับสร้างโมเดล Content-Based Recommendation
# โดยจะใช้ TF-IDF เพื่อแปลงข้อความเป็นเวกเตอร์ และใช้ KNN เพื่อหารายการที่คล้ายคลึงกัน
# พร้อมทั้งคำนวณคะแนนการมีส่วนร่วมจากคอมเมนต์และยอดเอนเกจเมนต์
def create_content_based_model(data, text_column='Content', comment_column='Comments', engagement_column='PostEngagement'):
    required_columns = [text_column, comment_column, engagement_column]
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"ข้อมูลขาดคอลัมน์ที่จำเป็น: {set(required_columns) - set(data.columns)}")
    train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
    tfidf = TfidfVectorizer(stop_words='english', max_features=6000, ngram_range=(1, 3), min_df=1, max_df=0.8)
    tfidf_matrix = tfidf.fit_transform(train_data[text_column].fillna(''))
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn.fit(tfidf_matrix)
    train_data['CommentCount'] = analyze_comments(train_data[comment_column])
    test_data['CommentCount'] = analyze_comments(test_data[comment_column])
    max_comment_count = train_data['CommentCount'].max()
    if max_comment_count > 0:
        train_data['NormalizedCommentCount'] = train_data['CommentCount'] / max_comment_count
        test_data['NormalizedCommentCount'] = test_data['CommentCount'] / max_comment_count
    else:
        train_data['NormalizedCommentCount'] = 0.0
        test_data['NormalizedCommentCount'] = 0.0
    train_data = normalize_engagement(train_data)
    train_data['NormalizedEngagement'] = normalize_scores(train_data[engagement_column])
    train_data['WeightedEngagement'] = train_data['NormalizedEngagement'] + train_data['NormalizedCommentCount']
    test_data = normalize_engagement(test_data)
    test_data['WeightedEngagement'] = test_data['NormalizedEngagement'] + test_data['NormalizedCommentCount']
    joblib.dump(tfidf, 'TFIDF_Model.pkl')
    joblib.dump(knn, 'KNN_Model.pkl')
    return tfidf, knn, train_data, test_data

# ฟังก์ชันนี้ใช้สำหรับสร้างโมเดล Collaborative Filtering ด้วยอัลกอริทึม SVD
# เพื่อทำนายคะแนนที่ผู้ใช้จะให้แก่โพสต์ โดยอิงจากพฤติกรรมของผู้ใช้รายอื่น
def create_collaborative_model(data, n_factors=150, n_epochs=70, lr_all=0.005, reg_all=0.5):
    required_columns = ['user_id', 'post_id']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"ข้อมูลขาดคอลัมน์ที่จำเป็น: {set(required_columns) - set(data.columns)}")
    melted_data = data.melt(id_vars=['user_id', 'post_id'], var_name='category', value_name='score')
    melted_data = melted_data[melted_data['score'] > 0]
    train_data, test_data = train_test_split(melted_data, test_size=0.25, random_state=42)
    reader = Reader(rating_scale=(melted_data['score'].min(), melted_data['score'].max()))
    trainset = Dataset.load_from_df(train_data[['user_id', 'post_id', 'score']], reader).build_full_trainset()
    model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
    model.fit(trainset)
    joblib.dump(model, 'Collaborative_Model.pkl')
    return model, test_data

# ฟังก์ชันนี้ใช้สำหรับให้คำแนะนำแบบ Hybrid โดยการรวมคะแนนจาก Collaborative Filtering, Content-Based และ Category
# เพื่อให้ได้คำแนะนำที่หลากหลายและตรงกับความสนใจของผู้ใช้มากที่สุด
def recommend_hybrid(user_id, all_posts_data, collaborative_model, knn, tfidf, categories, alpha=0.50, beta=0.20):
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha ต้องอยู่ในช่วง 0 ถึง 1")
    if not (0 <= beta <= 1):
        raise ValueError("Beta ต้องอยู่ในช่วง 0 ถึง 1")
    recommendations = []
    for _, post in all_posts_data.iterrows():
        post_id = post['post_id']
        collab_score = 0.5
        try:
            collab_score = collaborative_model.predict(user_id, post_id).est
        except ValueError:
            pass
        content_score = 0.0
        try:
            post_content = str(post['Content']) if pd.notna(post['Content']) else ''
            tfidf_vector = tfidf.transform([post_content])
            if knn._fit_X.shape[0] > 0:
                n_neighbors = min(20, knn._fit_X.shape[0])
                distances, indices = knn.kneighbors(tfidf_vector, n_neighbors=n_neighbors)
                if len(indices[0]) > 0:
                    content_score = np.mean([all_posts_data.iloc[i]['NormalizedEngagement'] for i in indices[0]])
        except Exception as e:
            print(f"ข้อผิดพลาดในการคํานวณ Content-Based score สําหรับ post_id {post_id}: {e}")
            content_score = 0.0
        category_score = 0.0
        if categories:
            for category in categories:
                if category in post.index and pd.notna(post[category]) and post[category] == 1:
                    category_score += 1
        if categories and len(categories) > 0:
            category_score /= len(categories)
        else:
            category_score = 0.0
        final_score = (alpha * float(collab_score)) + \
                      ((1 - alpha) * float(content_score)) + \
                      (beta * float(category_score))
        recommendations.append((post_id, final_score))
    recommendations_df = pd.DataFrame(recommendations, columns=['post_id', 'score'])
    recommendations_df['random_order'] = np.random.rand(len(recommendations_df))
    if not recommendations_df['score'].empty and recommendations_df['score'].nunique() > 1:
        recommendations_df['normalized_score'] = normalize_scores(recommendations_df['score'])
    else:
        recommendations_df['normalized_score'] = 0.5
    return recommendations_df.sort_values(by=['normalized_score', 'random_order'], ascending=[False, False])['post_id'].tolist()

# ฟังก์ชันนี้ใช้สำหรับจัดลำดับคำแนะนำ โดยจะแบ่งโพสต์เป็นกลุ่มๆ
# และจัดลำดับความสำคัญของโพสต์ที่ยังไม่เคยดู, โพสต์ที่แสดงไปแล้วแต่ยังไม่โต้ตอบ
# และโพสต์ที่เคยโต้ตอบแล้ว เพื่อให้คำแนะนำมีความสดใหม่และหลากหลาย
def split_and_rank_recommendations(recommendations, user_interactions, impression_history, total_posts_in_db):
    unique_recommendations_ids = [int(p) if isinstance(p, float) else p for p in list(dict.fromkeys(recommendations))]
    user_interactions_set = set([int(p) if isinstance(p, float) else p for p in user_interactions])
    impression_history_set = set([int(p) for entry in impression_history for p in [entry['post_id']]])

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

    group_A_not_recently_shown = [
        post_id for post_id in remaining_posts_to_mix
        if post_id not in impression_history_set
    ]

    group_B_recently_shown = [
        post_id for post_id in remaining_posts_to_mix
        if post_id in impression_history_set
    ]

    shuffled_remaining = []
    combined_for_shuffling = group_A_not_recently_shown + group_B_recently_shown

    num_to_demote_from_history = min(len(impression_history), int(len(impression_history) * 0.25))
    posts_to_demote = set([entry['post_id'] for entry in impression_history[:num_to_demote_from_history]])

    demoted_posts = []
    non_demoted_posts = []

    for post_id in unique_recommendations_ids:
        if post_id in posts_to_demote:
            demoted_posts.append(post_id)
        else:
            non_demoted_posts.append(post_id)

    truly_unviewed_non_demoted = [
        post_id for post_id in non_demoted_posts
        if post_id not in user_interactions_set and post_id not in impression_history_set
    ]

    remaining_non_demoted_and_not_truly_unviewed = [
        post_id for post_id in non_demoted_posts
        if post_id not in set(truly_unviewed_non_demoted)
    ]

    num_unviewed_first_current_run = min(30, len(truly_unviewed_non_demoted))
    final_recommendations_ordered.extend(truly_unviewed_non_demoted[:num_unviewed_first_current_run])

    remaining_to_shuffle_and_mix = truly_unviewed_non_demoted[num_unviewed_first_current_run:] + \
                                   remaining_non_demoted_and_not_truly_unviewed + \
                                   demoted_posts

    shuffled_segment = []
    block_size = 5

    blocks = [
        remaining_to_shuffle_and_mix[i : i + block_size]
        for i in range(0, len(remaining_to_shuffle_and_mix), block_size)
    ]

    for block in blocks:
        block_with_priority = []
        for post_id in block:
            priority_score = 0
            if post_id in posts_to_demote:
                priority_score = 2
            elif post_id in impression_history_set:
                priority_score = 1
            block_with_priority.append((post_id, priority_score))

        block_with_priority.sort(key=lambda x: (x[1], random.random()))

        shuffled_segment.extend([post_id for post_id, _ in block_with_priority])

    final_recommendations_ordered.extend(shuffled_segment)

    print("Top N Unviewed (prioritized):", final_recommendations_ordered[:num_unviewed_first_current_run])
    print("Remaining Overall Recommendations (block-shuffled, with recently shown and demoted pushed back):", final_recommendations_ordered[num_unviewed_first_current_run:])
    print("Final Recommendations (ordered, after mix):", final_recommendations_ordered)

    return final_recommendations_ordered

# ==================== SLIP & PROMPTPAY FUNCTIONS (from Slip.py) ====================

# ฟังก์ชันนี้ใช้ค้นหาข้อมูลคำสั่งซื้อ (Order) ด้วย ID ที่กำหนด
# เพื่อดึงรายละเอียดของคำสั่งซื้อนั้นๆ จากฐานข้อมูล
def find_order_by_id(order_id):
    order = Order.query.filter_by(id=order_id).first()
    if not order:
        return None
    return {
        'id': order.id,
        'user_id': order.user_id,
        'amount': order.amount,
        'status': order.status,
        'promptpay_qr_payload': order.promptpay_qr_payload,
        'slip_image': order.slip_image,
        'renew_ads_id': order.renew_ads_id,
        'package_id': order.package_id,
        'show_at': order.show_at
    }

# ฟังก์ชันนี้ใช้ค้นหาข้อมูลโฆษณา (Ad) ที่ผูกกับ Order ID ที่กำหนด
# เพื่อตรวจสอบสถานะและวันหมดอายุของโฆษณาที่เกี่ยวข้องกับคำสั่งซื้อ
def find_ad_by_order_id(order_id):
    ad = Ad.query.filter_by(order_id=order_id).first()
    if not ad:
        return None
    return {
        'id': ad.id,
        'status': ad.status,
        'expiration_date': ad.expiration_date,
        'show_at': ad.show_at
    }

# ฟังก์ชันนี้ใช้ค้นหาข้อมูลโฆษณา (Ad) ด้วย Ad ID ที่กำหนด
# เพื่อดึงรายละเอียดทั้งหมดของโฆษณาที่ต้องการต่ออายุหรือจัดการ
def find_ad_by_id(ad_id):
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

# ฟังก์ชันนี้ใช้ดึงจำนวนวันของแพ็กเกจโฆษณา (duration_days) จาก AdPackage
# เพื่อนำไปคำนวณวันหมดอายุของโฆษณา
def get_ad_package_duration(package_id):
    pkg = AdPackage.query.filter_by(package_id=package_id).first()
    if not pkg:
        print(f"❌ [ERROR] AdPackage with ID {package_id} not found.")
        return None
    return pkg.duration_days

# ฟังก์ชันนี้ใช้อัปเดตสถานะของคำสั่งซื้อ (Order) และบันทึกพาธของรูปสลิป
# เพื่อยืนยันการรับชำระเงินในฐานข้อมูล
def update_status_and_slip_info(order_id, new_status, slip_image_path, slip_transaction_id):
    order = Order.query.filter_by(id=order_id).first()
    if not order:
        print(f"❌ [ERROR] Order ID {order_id} not found for status update.")
        return False
    order.status = new_status
    order.slip_image = slip_image_path
    order.updated_at = datetime.now()
    try:
        db.session.commit()
        print(f"✅ Order ID: {order_id} status updated to '{new_status}' with slip info.")
        return True
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error updating order status for ID {order_id}: {e}")
        return False

# ฟังก์ชันนี้ใช้อัปเดตสถานะของโฆษณา (Ad) เท่านั้น
# โดยไม่กระทบกับวันหมดอายุหรือวันเริ่มต้นแสดงผลของโฆษณา
def update_ad_status(ad_id, new_status):
    ad = Ad.query.filter_by(id=ad_id).first()
    if not ad:
        print(f"❌ [ERROR] Ad ID {ad_id} not found for status update.")
        return False
    ad.status = new_status
    ad.updated_at = datetime.now()
    try:
        db.session.commit()
        print(f"✅ Ad ID: {ad_id} status updated to '{new_status}'.")
        return True
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error updating ad status for ID {ad_id}: {e}")
        return False

# ฟังก์ชันนี้ใช้อัปเดตสถานะและวันหมดอายุของโฆษณาสำหรับการต่ออายุ
# เพื่อขยายระยะเวลาการแสดงผลของโฆษณาเดิม
def update_ad_for_renewal(ad_id, new_status, new_expiration_date):
    ad = Ad.query.filter_by(id=ad_id).first()
    if not ad:
        print(f"❌ [ERROR] Ad ID {ad_id} not found for renewal update.")
        return False
    ad.status = new_status
    ad.expiration_date = new_expiration_date
    ad.updated_at = datetime.now()
    try:
        db.session.commit()
        print(f"✅ Ad ID: {ad_id} status updated to '{new_status}' and expiration date extended to {new_expiration_date.strftime('%Y-%m-%d')}.")
        return True
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error updating ad for renewal ID {ad_id}: {e}")
        return False

# ฟังก์ชันนี้ใช้บันทึก PromptPay QR Payload ลงในข้อมูลคำสั่งซื้อ (Order)
# เพื่อให้สามารถอ้างอิง QR Code ที่สร้างขึ้นได้ในอนาคต
def update_order_with_promptpay_payload_db(order_id, payload_to_store_in_db):
    order = Order.query.filter_by(id=order_id).first()
    if not order:
        print(f"❌ [ERROR] Order ID {order_id} not found for payload update.")
        return False
    order.promptpay_qr_payload = payload_to_store_in_db
    order.updated_at = datetime.now()
    try:
        db.session.commit()
        print(f"✅ Order ID: {order_id} updated with PromptPay payload.")
        return True
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error updating order with PromptPay payload: {e}")
        return False

# ฟังก์ชันนี้ใช้สร้างโฆษณา (Ad) ใหม่ในฐานข้อมูล
# โดยจะผูกกับคำสั่งซื้อที่ชำระเงินเรียบร้อยแล้ว
def create_advertisement_db(order_data):
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
    try:
        db.session.add(ad)
        db.session.commit()
        print(f"🚀 Advertisement ID: {ad.id} created for Order ID: {order_data['id']} with status 'paid'.")
        return ad.id
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error creating advertisement for Order ID {order_data['id']}: {e}")
        return None

# ฟังก์ชันนี้ใช้สร้าง PromptPay QR Code สำหรับคำสั่งซื้อที่กำหนด
# โดยจะตรวจสอบสถานะคำสั่งซื้อว่าพร้อมสำหรับการสร้าง QR หรือไม่
def generate_promptpay_qr_for_order(order_id):
    order = find_order_by_id(order_id)
    if not order:
        print(f"❌ [WARN] Order ID {order_id} not found for QR generation.")
        return {"success": False, "message": "ไม่พบคำสั่งซื้อ"}

    is_new_ad_approved = order["status"] == 'approved' and order.get("renew_ads_id") is None
    is_renewal_ad_pending = order["status"] == 'pending' and order.get("renew_ads_id") is not None

    if is_new_ad_approved or is_renewal_ad_pending:
        print(f"✅ [INFO] Order ID {order_id} is eligible for QR generation. Status: '{order['status']}', Renew Ad: {order.get('renew_ads_id')}.")
    else:
        log_message = f"❌ [WARN] Cannot generate QR for order {order_id}. Current status: '{order['status']}'."
        if order["status"] == 'pending' and order.get("renew_ads_id") is None:
            log_message += " (New ad order not yet approved by admin)."
            return {"success": False, "message": "ไม่สามารถสร้าง QR Code ได้ ต้องรอให้แอดมินอนุมัติเนื้อหาก่อน"}
        else:
            log_message += " (Invalid status for QR generation)."
            return {"success": False, "message": "ไม่สามารถสร้าง QR Code ได้ สถานะคำสั่งซื้อไม่ถูกต้อง"}
        print(log_message)

    amount = float(order["amount"])
    if promptpay_qrcode is None:
        print(f"❌ [ERROR] promptpay_qrcode library not found.")
        return {"success": False, "message": "ไม่พบไลบรารี promptpay กรุณาติดตั้งก่อน"}

    promptpay_id = os.getenv("PROMPTPAY_ID")
    if not promptpay_id:
        print(f"❌ [ERROR] PROMPTPAY_ID environment variable not set.")
        return {"success": False, "message": "ไม่พบ PromptPay ID ในการตั้งค่า"}

    original_scannable_payload = promptpay_qrcode.generate_payload(promptpay_id, amount)

    if not update_order_with_promptpay_payload_db(order_id, original_scannable_payload):
        print(f"❌ [ERROR] Failed to save QR Code payload to database for order {order_id}.")
        return {"success": False, "message": "ไม่สามารถบันทึกข้อมูล QR Code ลงฐานข้อมูลได้"}

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
    if hasattr(img, 'get_image'):
        img.get_image().save(buffered, "PNG")
    else:
        img.save(buffered, "PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return {"success": True, "message": "สร้าง QR Code สำเร็จ", "qrcode_base64": img_b64, "payload": original_scannable_payload}

# ฟังก์ชันนี้ใช้ตรวจสอบว่าคำสั่งซื้อ (Order) สามารถอัปโหลดสลิปได้หรือไม่
# โดยพิจารณาจากสถานะของคำสั่งซื้อว่าเป็นโฆษณาใหม่ที่ได้รับการอนุมัติ หรือโฆษณาต่ออายุที่รอชำระ
def can_upload_slip(order):
    if not order:
        return False

    is_new_ad_approved = order["status"] == 'approved' and order.get("renew_ads_id") is None
    is_renewal_ad_pending = order["status"] == 'pending' and order.get("renew_ads_id") is not None

    return is_new_ad_approved or is_renewal_ad_pending

# ฟังก์ชันนี้ใช้จัดรูปแบบวันที่ให้เป็นภาษาไทยและปีพุทธศักราช
# เพื่อให้แสดงผลวันที่ในรูปแบบที่เข้าใจง่ายสำหรับผู้ใช้
def format_thai_date(date_obj):
    if not isinstance(date_obj, (datetime, date, str)):
        return "วันที่ไม่ถูกต้อง"

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
                    return "วันที่ไม่ถูกต้อง"

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

# ฟังก์ชันนี้ใช้สร้างและบันทึกการแจ้งเตือนเมื่อสถานะโฆษณาเปลี่ยนแปลง
# เพื่อแจ้งให้ผู้ใช้ทราบถึงความคืบหน้าหรือการเปลี่ยนแปลงที่เกิดขึ้นกับโฆษณาของตน
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
                content = f"โฆษณาของคุณได้รับการต่ออายุ {duration_to_use} วันสำเร็จแล้ว โฆษณานี้ขยายหมดอายุเป็นวันที่ {formatted_expiration_date}"
            else:
                content = 'โฆษณาของคุณได้รับการอนุมัติขึ้นแสดงแล้ว'
        elif new_status == 'paid':
            content = 'โฆษณาของคุณชำระเงินเรียบร้อยแล้ว รอแอดมินตรวจสอบ'
        elif new_status == 'approved':
            content = 'โฆษณาของคุณได้รับการตรวจสอบแล้ว กรุณาโอนเงินเพื่อแสดงโฆษณา'
        elif new_status == 'rejected':
            content = f'โฆษณาของคุณถูกปฏิเสธ เหตุผล: {admin_notes or "-"}'
        elif new_status == 'expired':
            content = 'โฆษณาของคุณหมดอายุแล้ว'
        elif new_status == 'expiring_soon':
            content = 'โฆษณาของคุณจะหมดอายุในอีก 3 วัน กรุณาต่ออายุเพื่อการแสดงผลอย่างต่อเนื่อง'
        else:
            content = f'สถานะโฆษณาของคุณเปลี่ยนเป็น {new_status}'

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

# ฟังก์ชันนี้ใช้ตรวจสอบสลิปการโอนเงินและอัปเดตสถานะของคำสั่งซื้อและโฆษณา
# รองรับทั้งการชำระเงินสำหรับโฆษณาใหม่และการต่ออายุโฆษณา โดยจะเรียกใช้ SlipOK API เพื่อยืนยันความถูกต้องของสลิป
def verify_payment_and_update_status(order_id, slip_image_path, payload_from_client, db):
    print(f"\n--- Processing payment for Order ID: {order_id} ---")
    print(f"Slip image path: {slip_image_path}")
    print(f"Payload (from client - original QR data): {payload_from_client}")

    order = find_order_by_id(order_id)
    if not order:
        print(f"❌ [ERROR] Order ID {order_id} not found.")
        return {"success": False, "message": "ไม่พบคำสั่งซื้อ"}

    try:
        if not can_upload_slip(order):
            log_message = f"❌ [WARN] Cannot upload slip for Order ID {order_id}. Current status: {order.get('status')}."
            if order.get("status") == 'pending' and order.get("renew_ads_id") is None:
                log_message += " (New ad order not yet approved by admin)."
                print(log_message)
                return {"success": False, "message": "ไม่สามารถอัปโหลดสลิปได้ ต้องรอให้แอดมินอนุมัติเนื้อหาก่อน"}
            else:
                log_message += " (Invalid order status)."
                print(log_message)
                return {"success": False, "message": "ไม่สามารถอัปโหลดสลิปได้ สถานะคำสั่งซื้อไม่ถูกต้อง"}

        ad_related = None
        if order.get("renew_ads_id") is not None:
            ad_related = find_ad_by_id(order["renew_ads_id"])
            if not ad_related:
                print(f"❌ [ERROR] Associated ad for renewal (ID {order['renew_ads_id']}) not found for Order ID {order_id}.")
                return {"success": False, "message": "ไม่พบโฆษณาที่ต้องการต่ออายุ"}

            today = datetime.now().date()
            if isinstance(ad_related.get('expiration_date'), date) and ad_related['expiration_date'] < today:
                print(f"❌ [WARN] Cannot renew ad ID {ad_related['id']} for Order ID {order_id}. Ad has expired on {ad_related['expiration_date'].strftime('%Y-%m-%d')}.")
                return {"success": False, "message": "ไม่สามารถต่ออายุโฆษณาได้ เนื่องจากโฆษณาหมดอายุแล้ว"}

            if ad_related.get('status') not in ['active', 'expiring_soon', 'paused']:
                print(f"❌ [WARN] Associated ad ID {ad_related['id']} for Order ID {order_id} is not in a renewable status. Current ad status: {ad_related['status']}.")
                return {"success": False, "message": "โฆษณาสำหรับคำสั่งซื้อนี้ไม่อยู่ในสถานะที่สามารถต่ออายุได้"}
        else:
            ad_related = find_ad_by_order_id(order_id)
            if ad_related and ad_related.get('status') in ['active', 'rejected', 'paid']:
                print(f"❌ [WARN] Associated ad for Order ID {order_id} is already processed. Current ad status: {ad_related['status']}.")
                return {"success": False, "message": "โฆษณาสำหรับคำสั่งซื้อนี้มีการดำเนินการไปแล้ว"}

        SLIP_OK_API_ENDPOINT = os.getenv("SLIP_OK_API_ENDPOINT", "https://api.slipok.com/api/line/apikey/49130")
        SLIP_OK_API_KEY = os.getenv("SLIP_OK_API_KEY", "SLIPOKKBE52WN")

        if not os.path.exists(slip_image_path):
            print(f"❌ [ERROR] Slip image file not found at '{slip_image_path}'")
            return {"success": False, "message": "ไม่พบไฟล์รูปภาพสลิป"}

        response = None
        with open(slip_image_path, 'rb') as img_file:
            files = {'files': img_file}
            form_data_for_slipok = {
                'log': 'true',
                'amount': str(float(order["amount"]))
            }
            headers = {
                "x-authorization": SLIP_OK_API_KEY,
            }
            print(f"Sending request to SlipOK API: {SLIP_OK_API_ENDPOINT}")
            print(f"Headers sent: {headers}")
            print(f"Form Data sent to SlipOK: {form_data_for_slipok}")

            response = requests.post(SLIP_OK_API_ENDPOINT, files=files, data=form_data_for_slipok, headers=headers, timeout=30)
            response.raise_for_status()

            print(f"DEBUG: Full SlipOK response text: {response.text}")
            slip_ok_response_data = response.json()
            print(f"Received response from SlipOK: {slip_ok_response_data}")

            if not slip_ok_response_data.get("success"):
                error_message = slip_ok_response_data.get("message", "Unknown error from SlipOK API")
                print(f"❌ Log: Error from SlipOK API: {error_message}")
                return {"success": False, "message": f"การตรวจสอบสลิปไม่สำเร็จ: {error_message}"}

            slipok_data = slip_ok_response_data.get("data")
            if not slipok_data:
                print(f"❌ Log: Unexpected response format from SlipOK API: 'data' field is missing or empty.")
                return {"success": False, "message": "รูปแบบข้อมูลจากระบบตรวจสอบสลิปไม่ถูกต้อง (ไม่พบข้อมูลสลิป)"}

            slip_transaction_id_from_api = slipok_data.get("transRef")
            slip_amount = float(slipok_data.get("amount", 0.0))

            if not slip_transaction_id_from_api:
                print(f"❌ Log: Missing 'transRef' in SlipOK 'data' object.")
                return {"success": False, "message": "รูปแบบข้อมูลจากระบบตรวจสอบสลิปไม่ถูกต้อง (ไม่พบ Transaction ID)"}

        if abs(slip_amount - float(order.get("amount"))) > 0.01:
            print(f"❌ [WARN] Amount mismatch. Order: {order.get('amount')}, Slip: {slip_amount}")
            return {"success": False, "message": f"ยอดเงินไม่ถูกต้อง (ต้องการ {order.get('amount'):.2f} บาท แต่ได้รับ {slip_amount:.2f} บาท)"}

        if not update_status_and_slip_info(order_id, "paid", slip_image_path, slip_transaction_id_from_api):
            raise Exception("Failed to update order status and slip info.")

        ad_id_to_return = None
        ad_new_status_for_notification = None

        if order.get("renew_ads_id") is not None:
            current_ad = find_ad_by_id(order["renew_ads_id"])
            if not current_ad:
                raise Exception(f"Ad with ID {order['renew_ads_id']} not found for renewal processing after order update.")

            duration_days = get_ad_package_duration(order["package_id"])
            if duration_days is None:
                raise Exception(f"Ad package duration not found for package_id {order['package_id']} for renewal.")

            original_expiration = current_ad.get('expiration_date')

            renewal_start_date_candidate = None

            if original_expiration:
                if isinstance(original_expiration, date) and not isinstance(original_expiration, datetime):
                    renewal_start_date_candidate = datetime.combine(original_expiration, datetime.min.time())
                elif isinstance(original_expiration, datetime):
                    renewal_start_date_candidate = original_expiration

            if original_expiration:
                if renewal_start_date_candidate:
                    calculated_renewal_start = renewal_start_date_candidate + timedelta(days=1)
                else:
                    calculated_renewal_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

                if calculated_renewal_start.date() < datetime.now().date():
                    actual_renewal_start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                else:
                    actual_renewal_start_date = calculated_renewal_start

                new_expiration_date = actual_renewal_start_date + timedelta(days=duration_days - 1)

            else:
                order_show_at = order.get('show_at')
                actual_start_date_for_new_ad = None
                if order_show_at:
                    if isinstance(order_show_at, date) and not isinstance(order_show_at, datetime):
                        actual_start_date_for_new_ad = datetime.combine(order_show_at, datetime.min.time())
                    elif isinstance(order_show_at, datetime):
                        actual_start_date_for_new_ad = order_show_at

                if actual_start_date_for_new_ad is None:
                    actual_start_date_for_new_ad = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

                new_expiration_date = actual_start_date_for_new_ad + timedelta(days=duration_days - 1)

            if not update_ad_for_renewal(current_ad['id'], "active", new_expiration_date.date()):
                raise Exception("Failed to update existing ad status and expiration date for renewal.")

            ad_id_to_return = current_ad['id']
            ad_new_status_for_notification = 'active'

        else:
            ad_id = None
            ad = find_ad_by_order_id(order_id)
            if ad:
                ad_id = ad['id']
                if not update_ad_status(ad_id, "paid"):
                    raise Exception("Failed to update existing ad status to 'paid' for new ad.")
            else:
                ad_id = create_advertisement_db(order)
                if ad_id is None:
                    raise Exception("Failed to create new advertisement.")

            ad_id_to_return = ad_id
            ad_new_status_for_notification = 'paid'

        db.session.commit()
        print(f"✅ [INFO] Transaction committed successfully for Order ID: {order_id}.")

        if ad_id_to_return and ad_new_status_for_notification:
            notify_ads_status_change(db, ad_id_to_return, ad_new_status_for_notification)

        if order.get("renew_ads_id") is not None:
            duration_days = get_ad_package_duration(order["package_id"])
            message = f"ชำระเงินสำเร็จ! โฆษณาของคุณได้รับการต่ออายุเพิ่มอีก {duration_days} วันเรียบร้อยแล้ว"
        else:
            message = "ชำระเงินสำเร็จ! กรุณารอแอดมินตรวจสอบ"

        return {"success": True, "message": message, "ad_id": ad_id_to_return}

    except requests.exceptions.Timeout:
        try:
            db.session.rollback()
        except Exception as rollback_e:
            print(f"⚠️ [WARN] Error during rollback: {rollback_e}")
        print(f"❌ [API ERROR] SlipOK Timeout: API ไม่ตอบกลับภายในเวลาที่กำหนดสำหรับ Order ID: {order_id}.")
        return {"success": False, "message": "ระบบตรวจสอบสลิปตอบกลับช้าเกินไป โปรดลองอีกครั้ง"}
    except requests.exceptions.HTTPError as e:
        try:
            db.session.rollback()
        except Exception as rollback_e:
            print(f"⚠️ [WARN] Error during rollback: {rollback_e}")

        slipok_error_message = "ไม่ทราบข้อผิดพลาดจาก SlipOK API"
        if response is not None:
            try:
                error_details = response.json()
                slipok_error_message = error_details.get('message', 'ไม่พบข้อความผิดพลาดจาก SlipOK')
                print(f"❌ [API ERROR] HTTP Error {e.response.status_code} for Order ID: {order_id}. SlipOK Message: {slipok_error_message}. URL: {e.request.url}")
                print(f"    Full SlipOK Response Body: {response.text}")
            except Exception as json_e:
                print(f"❌ [API ERROR] HTTP Error {e.response.status_code} for Order ID: {order_id}. URL: {e.request.url}. Cannot parse SlipOK response as JSON. Error: {json_e}")
                print(f"    Full SlipOK Response Text (Non-JSON): {response.text}")
                slipok_error_message = f"เกิดข้อผิดพลาดในการประมวลผลคำตอบจาก SlipOK: {response.text[:100]}..."
        else:
            print(f"❌ [API ERROR] HTTP Error for Order ID: {order_id}. Error: {e}. (No SlipOK response object found)")

        return {"success": False, "message": f"การตรวจสอบสลิปไม่สำเร็จ: {slipok_error_message}"}

    except requests.exceptions.RequestException as e:
        try:
            db.session.rollback()
        except Exception as rollback_e:
            print(f"⚠️ [WARN] Error during rollback: {rollback_e}")
        print(f"❌ [API ERROR] Connection Error: ไม่สามารถเชื่อมต่อกับ SlipOK API สำหรับ Order ID: {order_id}. Error: {e}")
        return {"success": False, "message": f"เกิดข้อผิดพลาดในการเชื่อมต่อกับระบบตรวจสอบสลิป: {e}"}
    except ValueError:
        try:
            db.session.rollback()
        except Exception as rollback_e:
            print(f"⚠️ [WARN] Error during rollback: {rollback_e}")
        print(f"❌ [API ERROR] Data Parsing Error: ไม่สามารถอ่านยอดเงินจาก SlipOK response ได้สำหรับ Order ID: {order_id}.")
        return {"success": False, "message": "รูปแบบยอดเงินจากระบบตรวจสอบสลิปไม่ถูกต้อง"}
    except Exception as e:
        try:
            if db and hasattr(db, 'session'):
                db.session.rollback()
        except Exception as rollback_e:
            print(f"⚠️ [WARN] Error during rollback: {rollback_e}")

        print(f"❌ [APP ERROR] Transaction failed for Order ID: {order_id}. Rolling back changes. Error: {e}")
        return {"success": False, "message": f"เกิดข้อผิดพลาดในการทำรายการ: {e}"}

# ==================== FLASK ROUTES ====================

# NSFW Detection Route
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

        # ตรวจสอบ user_id: ต้องมี user_id ถึงจะสร้างโพสต์ได้งับ
        if not user_id:
            return jsonify({"error": "You are not authorized to create a post for this user"}), 403

        # เริ่มต้นประมวลผลรูปภาพที่อัปโหลดงับ
        photo_urls = []
        invalid_photos = []
        
        print(f"Processing {len(photos)} photos...")
        
        for photo in photos:
            if not photo or not photo.filename:
                continue
                
            try:
                filename = secure_filename(photo.filename)
                photo_path = os.path.join(UPLOAD_FOLDER, filename)
                photo.save(photo_path)
                
                print(f"Processing photo: {filename}")
                is_nude, result = nude_predict_image(photo_path)
                
                if is_nude:
                    print(f"NSFW detected in {filename}")
                    # ลบไฟล์ภาพที่ไม่เหมาะสมที่ถูกตรวจพบงับ
                    if os.path.exists(photo_path):
                        os.remove(photo_path)
                    
                    # เก็บข้อมูลภาพที่ไม่เหมาะสมเพื่อนแจ้งกลับไปที่ Client งับ
                    invalid_photos.append({
                        "filename": filename,
                        "reason": "พบภาพโป๊ (Hentai หรือ Pornography > 20%)",
                        "details": result
                    })
                else:
                    print(f"Photo {filename} is safe")
                    photo_urls.append(f'/uploads/{filename}')
                    
            except Exception as e:
                print(f"Error processing photo {photo.filename}: {e}")
                # หากประมวลผลผิดพลาด ให้ลบไฟล์แล้วแจ้งเตือนงับ
                if 'photo_path' in locals() and os.path.exists(photo_path):
                    os.remove(photo_path)
                invalid_photos.append({
                    "filename": photo.filename,
                    "reason": "ไม่สามารถประมวลผลภาพได้",
                    "details": {"error": str(e)}
                })
        
        # หากมีภาพที่ไม่เหมาะสม ให้แจ้งเตือนและยกเลิกการสร้างโพสต์งับ
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

        # ประมวลผล URL ของวิดีโอที่อัปโหลดงับ
        video_urls = []
        for video in videos:
            if not video or not video.filename:
                continue
            filename = secure_filename(video.filename)
            video_path = os.path.join(UPLOAD_FOLDER, filename)
            video.save(video_path)
            video_urls.append(f'/uploads/{filename}')

        photo_urls_json = json.dumps(photo_urls)
        video_urls_json = json.dumps(video_urls)

        # เริ่มบันทึกข้อมูลลงฐานข้อมูล MySQL งับ
        try:
            # สร้าง SQL query สำหรับเพิ่มโพสต์ใหม่งับ
            insert_query = text("""
                INSERT INTO posts (user_id, Title, content, ProductName, CategoryID, photo_url, video_url, status, updated_at)
                VALUES (:user_id, :title, :content, :product_name, :category_id, :photo_urls, :video_urls, 'active', NOW())
            """)
            
            # รัน query งับ
            result = db.session.execute(insert_query, {
                'user_id': user_id,
                'title': title,
                'content': content,
                'product_name': product_name,
                'category_id': category, 
                'photo_urls': photo_urls_json,
                'video_urls': video_urls_json
            })
            
            # Commit การเปลี่ยนแปลงลงฐานข้อมูลงับ
            db.session.commit()
            
            # ดึง post_id ที่เพิ่งสร้างงับ
            post_id = result.lastrowid
            
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
            return jsonify({"error": "ไม่สามารถบันทึกโพสต์ลงฐานข้อมูลได้"}), 500

    except Exception as error:
        print("Internal server error:", str(error))
        return jsonify({"error": "Internal server error"}), 500

@app.route('/ai/posts/<int:id>', methods=['PUT'])
def update_post(id):
    try:
        Title = request.form.get('Title')
        content = request.form.get('content')
        ProductName = request.form.get('ProductName')
        CategoryID = request.form.get('CategoryID')
        user_id = request.form.get('user_id')
        
        # รับรูปภาพและวิดีโอที่มีอยู่เดิม (ถ้ามี) จาก JSON string งับ
        existing_photos_str = request.form.get('existing_photos')
        existing_videos_str = request.form.get('existing_videos')
        
        # แปลง JSON string เป็น Python list งับ
        existing_photos = json.loads(existing_photos_str) if existing_photos_str else []
        existing_videos = json.loads(existing_videos_str) if existing_videos_str else []
        
        photos = request.files.getlist('photo')
        videos = request.files.getlist('video')

        if not user_id:
            return jsonify({"error": "You are not authorized to update this post"}), 403

        # รวมรูปภาพเดิมและรูปภาพใหม่ที่อัปโหลดงับ
        photo_urls = existing_photos if existing_photos else []
        invalid_photos = []

        for photo in photos:
            photo_path = None # กำหนด photo_path ไว้ก่อนเริ่ม try block งับ
            if not photo or not photo.filename:
                continue
                
            try:
                filename = secure_filename(photo.filename)
                photo_path = os.path.join(UPLOAD_FOLDER, filename)
                photo.save(photo_path) # บันทึกไฟล์ก่อนตรวจสอบ AI งับ

                print(f"Processing new photo for update: {filename}")
                is_nude, result = nude_predict_image(photo_path)
                
                if is_nude:
                    print(f"NSFW detected in {filename} during update")
                    # ลบไฟล์ภาพที่ไม่เหมาะสมทันทีงับ
                    if os.path.exists(photo_path):
                        os.remove(photo_path)
                    
                    invalid_photos.append({
                        "filename": filename,
                        "reason": "พบภาพโป๊ (Hentai หรือ Pornography > 20%)",
                        "details": result
                    })
                else:
                    print(f"New photo {filename} is safe")
                    photo_urls.append(f'/uploads/{filename}')
                                        
            except Exception as e:
                print(f"Error processing photo {photo.filename} during update: {e}")
                # หากเกิดข้อผิดพลาด ให้ลบไฟล์แล้วแจ้งเตือนงับ
                if photo_path and os.path.exists(photo_path): 
                    os.remove(photo_path)
                invalid_photos.append({
                    "filename": photo.filename,
                    "reason": "ไม่สามารถประมวลผลภาพได้",
                    "details": {"error": str(e)}
                })
        
        # หากมีภาพที่ไม่เหมาะสม ให้แจ้งเตือนและไม่อัปเดตโพสต์งับ
        if invalid_photos:
            print("แจ้งเตือน user: พบภาพไม่เหมาะสมในระหว่างการอัปเดต")
            return jsonify({
                "status": "warning",
                "message": "กรุณาเปลี่ยนภาพแล้วลองใหม่อีกครั้ง",
                "invalid_photos": invalid_photos,
                "valid_photos": photo_urls
            }), 400

        # รวมวิดีโอเดิมและวิดีโอใหม่ที่อัปโหลดงับ
        video_urls = existing_videos if existing_videos else []
        for video in videos:
            if not video or not video.filename:
                continue
            filename = secure_filename(video.filename)
            video_path = os.path.join(UPLOAD_FOLDER, filename)
            video.save(video_path)
            video_urls.append(f'/uploads/{filename}')

        photo_urls_json = json.dumps(photo_urls)
        video_urls_json = json.dumps(video_urls)

        # เริ่มอัปเดตข้อมูลในฐานข้อมูล MySQL งับ
        try:
            # สร้าง SQL query สำหรับอัปเดตโพสต์งับ
            update_query = text("""
                UPDATE posts 
                SET Title = :title, content = :content, ProductName = :product_name, 
                    CategoryID = :category_id, photo_url = :photo_urls, video_url = :video_urls, 
                    updated_at = NOW()
                WHERE id = :post_id AND user_id = :user_id
            """)
            
            # รัน query งับ
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
            
            # Commit การเปลี่ยนแปลงงับ
            db.session.commit()
            
            # ตรวจสอบว่ามีแถวที่ถูกอัปเดตหรือไม่ ถ้าไม่มีแปลว่าไม่พบโพสต์หรืองับ
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
            return jsonify({"error": "ไม่สามารถอัปเดตโพสต์ลงฐานข้อมูลได้"}), 500

    except Exception as error:
        print("Internal server error:", str(error))
        return jsonify({"error": "Internal server error"}), 500


@app.route("/ai/users/<int:userId>/profile", methods=['PUT'])
# @verifyToken # อย่าลืมเปิดใช้งาน middleware นี้ด้วยนะงับ
def update_user_profile(userId):
    try:
        username = request.form.get('username')
        bio = request.form.get('bio')
        gender = request.form.get('gender')
        birthday_str = request.form.get('birthday')

        profile_image_file = request.files.get('profileImage')

        if not all([username, bio, gender, birthday_str]):
            return jsonify({
                "error": "กรุณากรอกข้อมูลให้ครบทุกช่องงับ: ชื่อผู้ใช้, ไบโอ, เพศ, และวันเกิด"
            }), 400

        # จัดการและตรวจสอบรูปแบบวันเกิดงับ
        birthday = None
        date_formats = [
            "%Y-%m-%d",    # รูปแบบหลักจาก Front-end งับ
            "%d/%m/%Y",    # เผื่อไว้สำหรับรูปแบบอื่น ๆ งับ
            "%m/%d/%Y",
            "%Y/%m/%d",
            "%d-%m-%Y",
        ]
        
        # ลอง parse แบบ ISO standard ที่ไม่ขึ้นกับ locale งับ
        try:
            birthday = datetime.fromisoformat(birthday_str).strftime("%Y-%m-%d")
        except ValueError:
            # ถ้าจาก fromisoformat ไม่ได้ผล ให้ลองวิธีอื่น ๆ งับ
            for fmt in date_formats:
                try:
                    birthday = datetime.strptime(birthday_str, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue

        if birthday is None:
            return jsonify({"error": "รูปแบบวันเกิดไม่ถูกต้อง โปรดระบุในรูปแบบ YYYY-MM-DD"}), 400
        
        # จัดการและตรวจสอบรูปภาพโปรไฟล์ด้วย AI งับ
        profile_image_path = None
        if profile_image_file and profile_image_file.filename:
            try:
                filename = secure_filename(profile_image_file.filename)
                temp_image_path = os.path.join(UPLOAD_FOLDER, filename)
                profile_image_file.save(temp_image_path)

                print(f"Processing new profile image: {filename}")
                is_nude, result = nude_predict_image(temp_image_path)

                if is_nude:
                    print(f"NSFW detected in profile image {filename}")
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
                    return jsonify({
                        "status": "warning",
                        "message": "กรุณาเปลี่ยนภาพแล้วลองใหม่อีกครั้ง",
                        "details": result
                    }), 400
                else:
                    print(f"Profile image {filename} is safe")
                    profile_image_path = f'/uploads/{filename}'
            except Exception as e:
                print(f"Error processing profile image {profile_image_file.filename}: {e}")
                if temp_image_path and os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                return jsonify({
                    "error": "ไม่สามารถประมวลผลภาพโปรไฟล์ได้",
                    "details": str(e)
                }), 500

        # ตรวจสอบว่า username ซ้ำกับคนอื่นไหมงับ (ยกเว้นตัวเอง)
        check_username_query = text("""
            SELECT id FROM users WHERE username = :username AND id != :user_id
        """)
        check_results = db.session.execute(check_username_query, {
            'username': username,
            'user_id': userId
        }).fetchall()

        if len(check_results) > 0:
            return jsonify({"error": "ชื่อผู้ใช้นี้มีคนใช้แล้วงับ"}), 400

        # สร้าง SQL query สำหรับอัปเดตโปรไฟล์งับ
        update_profile_query = """
            UPDATE users SET username = :username, bio = :bio, gender = :gender, birthday = :birthday
        """
        update_data = {
            'username': username,
            'bio': bio,
            'gender': gender,
            'birthday': birthday,
            'user_id': userId
        }

        # ถ้ามีรูปโปรไฟล์ใหม่ ก็เพิ่มใน query งับ
        if profile_image_path:
            update_profile_query += ", picture = :picture"
            update_data['picture'] = profile_image_path

        update_profile_query += " WHERE id = :user_id"

        try:
            # รัน query และ commit การเปลี่ยนแปลงงับ
            result = db.session.execute(text(update_profile_query), update_data)
            db.session.commit()

            if result.rowcount == 0:
                return jsonify({"error": "ไม่พบผู้ใช้หรือไม่มีการเปลี่ยนแปลงข้อมูลงับ"}), 404

            return jsonify({
                "message": "อัปเดตโปรไฟล์สำเร็จงับ",
                "profileImage": profile_image_path if profile_image_path else "No new image uploaded",
                "username": username,
                "bio": bio,
                "gender": gender,
                "birthday": birthday
            }), 200

        except Exception as db_error:
            print(f"Database error during profile update: {db_error}")
            db.session.rollback()
            return jsonify({"error": "เกิดข้อผิดพลาดฐานข้อมูลขณะอัปเดตโปรไฟล์ผู้ใช้"}), 500

    except Exception as error:
        print(f"Internal server error: {error}")
        return jsonify({"error": "เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์"}), 500

# Recommendation Route
@app.route('/ai/recommend', methods=['POST'])
@verify_token
def recommend():
    try:
        user_id = request.user_id
        now = datetime.now()

        # ตรวจสอบว่ามีการร้องขอ refresh หรือไม่ (จาก client) งับ
        refresh_requested = request.args.get('refresh', 'false').lower() == 'true'

        if refresh_requested:
            # ลบข้อมูล cache สำหรับ user นี้ เพื่อให้ระบบคำนวณใหม่ งับ
            if user_id in recommendation_cache:
                del recommendation_cache[user_id]
            print(f"Cache for user_id: {user_id} invalidated due to client-side refresh request. Impression history RETAINED.") 

        # ตรวจสอบ Cache ปกติ ถ้าข้อมูลยังสดใหม่ ก็ใช้จาก cache เลยงับ
        if user_id in recommendation_cache and not refresh_requested: 
            cached_data, cache_timestamp = recommendation_cache[user_id]
            if (now - cache_timestamp).total_seconds() < (CACHE_EXPIRY_TIME_SECONDS / 2):
                print(f"Returning VERY FRESH cached recommendations for user_id: {user_id}")
                return jsonify(cached_data)
            print(f"Cached recommendations for user_id: {user_id} are still valid but slightly old. Recalculating for freshness.")

        # โหลดข้อมูลที่จำเป็นสำหรับการแนะนำโพสต์จากฐานข้อมูลงับ
        content_based_data, collaborative_data = load_data_from_db()

        try:
            # โหลดโมเดล AI ที่ใช้ในการแนะนำงับ
            knn = joblib.load('KNN_Model.pkl')
            collaborative_model = joblib.load('Collaborative_Model.pkl')
            tfidf = joblib.load('TFIDF_Model.pkl')
            
            # ตรวจสอบและประมวลผลข้อมูล Engagement และ Comment ก่อนส่งให้โมเดลใช้ งับ
            if 'NormalizedEngagement' not in content_based_data.columns:
                content_based_data = normalize_engagement(content_based_data, user_column='owner_id', engagement_column='PostEngagement')
            
            if 'CommentCount' not in content_based_data.columns:
                content_based_data['CommentCount'] = analyze_comments(content_based_data['Comments'])
                max_comment_count_for_normalization = content_based_data['CommentCount'].max()
                if max_comment_count_for_normalization > 0:
                    content_based_data['NormalizedCommentCount'] = content_based_data['CommentCount'] / max_comment_count_for_normalization
                else:
                    content_based_data['NormalizedCommentCount'] = 0.0

            if 'WeightedEngagement' not in content_based_data.columns:
                content_based_data['WeightedEngagement'] = content_based_data['NormalizedEngagement'] + content_based_data['NormalizedCommentCount']

        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
            return jsonify({"error": "Model files not found"}), 500

        # กำหนดหมวดหมู่ของสินค้าที่ใช้ในการแนะนำงับ
        categories = [
            'Electronics_Gadgets', 'Furniture', 'Outdoor_Gear', 'Beauty_Products', 'Accessories'
        ]

        # คำนวณคำแนะนำแบบ Hybrid (Content-based + Collaborative Filtering) งับ
        recommendations = recommend_hybrid(
            user_id,
            content_based_data,
            collaborative_model,
            knn,
            tfidf,
            categories,
            alpha=0.8,
            beta=0.2
        )

        if not recommendations:
            return jsonify({"error": "No recommendations found"}), 404

        # ดึงประวัติการมีปฏิสัมพันธ์ของ user เพื่อกรองโพสต์ที่ไม่ควรแนะนำซ้ำ งับ
        user_interactions = collaborative_data[collaborative_data['user_id'] == user_id]['post_id'].tolist()
        
        # ดึงประวัติการเห็นโพสต์ (impression history) ของ user งับ
        current_impression_history = impression_history_cache.get(user_id, [])
        print(f"Current Impression History for user {user_id}: {[entry['post_id'] for entry in current_impression_history]}")

        # กรองและจัดอันดับคำแนะนำสุดท้าย พร้อมส่งจำนวนโพสต์ทั้งหมดใน DB ด้วยงับ
        final_recommendations_ids = split_and_rank_recommendations(
            recommendations, 
            user_interactions, 
            current_impression_history,
            len(content_based_data) 
        )
        
        # สร้าง SQL query เพื่อดึงข้อมูลโพสต์ที่แนะนำจาก DB งับ
        placeholders = ', '.join([f':id_{i}' for i in range(len(final_recommendations_ids))])
        query = text(f"""
            SELECT posts.*, users.username, users.picture,
                   (SELECT COUNT(*) FROM likes WHERE post_id = posts.id AND user_id = :user_id) AS is_liked
            FROM posts 
            JOIN users ON posts.user_id = users.id
            WHERE posts.status = 'active' AND posts.id IN ({placeholders})
        """)

        params = {'user_id': user_id, **{f'id_{i}': post_id for i, post_id in enumerate(final_recommendations_ids)}}
        result = db.session.execute(query, params).fetchall()
        posts = [row._mapping for row in result]

        # จัดเรียงโพสต์ที่ได้จาก DB ตามลำดับที่ AI แนะนำงับ
        sorted_posts = sorted(posts, key=lambda x: final_recommendations_ids.index(x['id']))

        output = []
        for post in sorted_posts:
            output.append({
                "id": post['id'],
                "userId": post['user_id'],
                "title": post['Title'],
                "content": post['content'],
                "updated": post['updated_at'].astimezone(timezone.utc).replace(microsecond=0).isoformat() + 'Z',
                "photo_url": json.loads(post.get('photo_url', '[]')),
                "video_url": json.loads(post.get('video_url', '[]')),
                "userName": post['username'],
                "userProfileUrl": post['picture'],
                "is_liked": post['is_liked'] > 0
            })

        # เก็บผลลัพธ์ลง Cache งับ
        recommendation_cache[user_id] = (output, now)

        # อัปเดต Impression History ของ user งับ
        if user_id not in impression_history_cache:
            impression_history_cache[user_id] = []
        
        new_impressions_to_add = []
        current_impressions_id_set = {entry['post_id'] for entry in impression_history_cache[user_id]}
        for post_id in final_recommendations_ids:
            if post_id not in current_impressions_id_set:
                new_impressions_to_add.append({'post_id': post_id, 'timestamp': now})
        
        impression_history_cache[user_id].extend(new_impressions_to_add)

        impression_history_cache[user_id] = sorted(
            impression_history_cache[user_id], 
            key=lambda x: x['timestamp'], 
            reverse=True
        )[:IMPRESSION_HISTORY_MAX_ENTRIES]
        
        print(f"Updated Impression History for user {user_id}: {[entry['post_id'] for entry in impression_history_cache[user_id]]}")

        return jsonify(output)

    except KeyError as e:
        print(f"KeyError in recommend function: {e}")
        return jsonify({"error": f"KeyError: {e}"}), 500
    except Exception as e:
        print("Error in recommend function:", e)
        return jsonify({"error": "Internal Server Error"}), 500


@app.route('/api/generate-qrcode/<int:order_id>', methods=['GET'])
def api_generate_qrcode(order_id):
    """
    API สำหรับสร้าง PromptPay QR Code สำหรับคำสั่งซื้อ.
    จะสร้าง QR Code ได้หากคำสั่งซื้อเป็นโฆษณาใหม่ที่ 'approved'
    หรือเป็นคำสั่งซื้อต่ออายุที่ 'pending'.
    """
    # เรียกใช้ฟังก์ชัน generate_promptpay_qr_for_order ที่ได้รับการปรับแก้แล้วงับ
    result = generate_promptpay_qr_for_order(order_id)
    if not result['success']:
        # หากไม่สำเร็จ จะส่งข้อความผิดพลาดและสถานะ HTTP 400 งับ
        return jsonify(result), 400
    
    # หากสำเร็จ จะส่งข้อมูล QR Code และ payload กลับไปงับ
    return jsonify({
        'success': True,
        'order_id': order_id,
        'qrcode_base64': result.get('qrcode_base64'),
        'promptpay_payload': result.get('payload')
    })


@app.route('/api/verify-slip/<int:order_id>', methods=['POST'])
def api_verify_slip(order_id):
    """
    API สำหรับตรวจสอบสลิปการโอนเงินและอัปเดตสถานะคำสั่งซื้อ/โฆษณา.
    จะรองรับทั้งการชำระเงินสำหรับโฆษณาใหม่และการต่ออายุโฆษณา.
    """
    if 'slip_image' not in request.files:
        print(f"❌ [WARN] API Verify Slip: No 'slip_image' file found in request for order ID {order_id}.")
        return jsonify({'success': False, 'message': 'กรุณาอัปโหลดไฟล์สลิปการโอนเงิน'}), 400
    
    file = request.files['slip_image']
    if file.filename == '':
        print(f"❌ [WARN] API Verify Slip: Empty filename for 'slip_image' for order ID {order_id}.")
        return jsonify({'success': False, 'message': 'ไม่ได้เลือกไฟล์สลิป'}), 400
    
    if 'payload' not in request.form:
        print(f"❌ [WARN] API Verify Slip: No 'payload' (QR Code data) found in request form for order ID {order_id}.")
        return jsonify({'success': False, 'message': 'ต้องระบุ payload (ข้อมูล QR Code ที่สร้าง) เพื่อตรวจสอบสลิป'}), 400
    
    payload = request.form['payload']

    order = find_order_by_id(order_id)
    if not order:
        print(f"❌ [WARN] API Verify Slip: Order ID {order_id} not found.")
        return jsonify({'success': False, 'message': 'ไม่พบคำสั่งซื้อนี้'}), 404

    if not can_upload_slip(order):
        order_status = order.get('status', 'N/A') 
        renew_ad_id = order.get('renew_ads_id', 'N/A') 
        print(f"❌ [WARN] API Verify Slip: Order ID {order_id} not eligible for slip upload. Current status: {order_status}. Renew Ad: {renew_ad_id}.")
        return jsonify({'success': False, 'message': 'ไม่สามารถอัปโหลดสลิปได้ เนื่องจากสถานะคำสั่งซื้อไม่ถูกต้อง หรือยังไม่ได้รับการอนุมัติ'}), 400

    # สร้างชื่อไฟล์ที่ไม่ซ้ำกัน และบันทึกสลิปงับ
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    slip_dir = 'Slip'
    if not os.path.exists(slip_dir):
        os.makedirs(slip_dir)
    save_path = os.path.join(slip_dir, unique_filename)
    file.save(save_path)
    
    print(f"✅ [INFO] API Verify Slip: Slip image uploaded to {save_path} for Order ID {order_id}.")
    print(f"✅ [INFO] API Verify Slip: Payload from client (QR Code data): {payload}.")

    # เรียกใช้ฟังก์ชันตรวจสอบการชำระเงินและอัปเดตสถานะงับ
    result = verify_payment_and_update_status(order_id, save_path, payload, db)

    if not result.get('success'):
        return jsonify(result), 400

    return jsonify(result), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)