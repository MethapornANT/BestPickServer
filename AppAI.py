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

# โหลดโมเดลและ processor สำหรับ NSFW detection
MODEL_NAME = "strangerguardhf/nsfw_image_detection"
model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# mapping id เป็น label
id2label = {
    "0": "Anime Picture",
    "1": "Hentai",
    "2": "Normal",
    "3": "Pornography",
    "4": "Enticing or Sensual"
}

def nude_predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
        hentai_score = probs[1] * 100
        porn_score = probs[3] * 100
        
        # Debug logging
        print(f"NSFW Detection for {image_path}:")
        print(f"  Hentai: {hentai_score:.2f}%")
        print(f"  Pornography: {porn_score:.2f}%")
        print(f"  Is NSFW: {hentai_score > 20 or porn_score > 20}")
        
        return hentai_score > 20 or porn_score > 20, {id2label[str(i)]: round(probs[i]*100, 2) for i in range(len(probs))}
    except Exception as e:
        print(f"Error in NSFW detection for {image_path}: {e}")
        # หากเกิดข้อผิดพลาดในการตรวจสอบ ให้ถือว่าไม่ใช่ภาพโป๊ (เพื่อไม่ให้ระบบล่ม)
        return False, {"error": "ไม่สามารถตรวจสอบภาพได้ กรุณาลองใหม่อีกครั้ง"}

# ==================== WEB SCRAPING SETUP ====================
# สร้าง Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--window-size=1920x1080")
chrome_options.add_argument("--log-level=3")
chrome_options.binary_location = r"C:\chrome-win64\chrome.exe"  # ✅ Browser
chrome_driver_path = r"C:\chromedriver-win64\chromedriver.exe"  # ✅ Driver
chrome_service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

# ==================== DATABASE SETUP ====================
# Configure your database URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:1234@localhost/bestpick'

# Initialize the SQLAlchemy object
db = SQLAlchemy(app)

# ==================== SQLAlchemy Models สำหรับ Slip/Order/Ad ====================

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
    # --- ลบบรรทัดนี้ออก ---
    # slip_transaction_id = db.Column(db.String(255), nullable=True) 

    created_at = db.Column(db.DateTime, default=datetime.now) 
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now) 
    
    show_at = db.Column(db.Date, nullable=True) 

    def __repr__(self):
        return f'<Order {self.id}>'

# Ad และ AdPackage Models ที่เหลือคงเดิม
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

class AdPackage(db.Model):
    __tablename__ = 'ad_packages'
    package_id = db.Column(db.Integer, primary_key=True) 
    name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Numeric(10, 2), nullable=False)
    duration_days = db.Column(db.Integer, nullable=False) 

    def __repr__(self):
        return f'<AdPackage {self.package_id}>'


load_dotenv()
# Secret key for encoding/decoding JWT tokens
JWT_SECRET = os.getenv('JWT_SECRET')

# ==================== WEB SCRAPING FUNCTIONS ====================
# Filter products by name to match search term
def filter_products_by_name(products, search_name):
    filtered_products = []
    search_name_lower = search_name.lower()
    for product in products:
        product_name_lower = product['name'].lower()
        if re.search(search_name_lower, product_name_lower):
            filtered_products.append(product)
    return filtered_products[:1] if filtered_products else products[:1]

# Search and scrape Advice products
def search_and_scrape_advice_product(product_name, results):
    try:
        search_url = f"https://www.advice.co.th/search?keyword={product_name.replace(' ', '%20')}"
        driver.get(search_url)
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # ปรับการดึงข้อมูลสินค้าให้เฉพาะเจาะจงมากขึ้น
        product_divs = soup.find_all('div', {'class': 'item'})  
        products = []
        for product_div in product_divs:
            product_name = product_div.get('item-name')
            
            # เงื่อนไขตรวจสอบว่าในชื่อสินค้าต้องมีคำว่า "iPhone" และ "15 Pro"
            if product_name and "iphone" in product_name.lower() and "15 pro" in product_name.lower():
                price_tag = product_div.find('div', {'class': 'sales-price sales-price-font'})
                product_price = price_tag.text.strip() if price_tag else "Price not found"
                product_url = product_div.find('a', {'class': 'product-item-link'})['href']
                products.append({"name": product_name, "price": product_price, "url": product_url})
                
        # กรองข้อมูลสินค้าให้ได้เฉพาะสินค้าที่ตรงกับคำค้นหามากที่สุด
        results['Advice'] = filter_products_by_name(products, product_name) if products else [{"name": "Not found", "price": "-", "url": "#"}]
    except Exception as e:
        results['Advice'] = f"Error occurred during Advice scraping: {e}"

# Scrape JIB
def search_and_scrape_jib_product_from_search(product_name, results):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        search_url = f"https://www.jib.co.th/web/product/product_search/0?str_search={product_name.replace(' ', '%20')}"
        response = requests.get(search_url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            product_containers = soup.find_all('div', {'class': 'divboxpro'})
            products = []
            for product_container in product_containers:
                product_name_tag = product_container.find('span', {'class': 'promo_name'})
                found_product_name = product_name_tag.text.strip() if product_name_tag else "Product name not found"
                if re.search(product_name.lower(), found_product_name.lower()):  # Check for matching name
                    price_tag = product_container.find('p', {'class': 'price_total'})
                    product_price = price_tag.text.strip() + " บาท" if price_tag else "Price not found"
                    productsearch = product_container.find('div', {'class': 'row size_img center'})
                    product_url = productsearch.find('a')['href']
                    products.append({"name": found_product_name, "price": product_price, "url": product_url})
            results['JIB'] = filter_products_by_name(products, product_name)
        else:
            results['JIB'] = f"Failed to search JIB. Status code: {response.status_code}"
    except Exception as e:
        results['JIB'] = f"Error occurred during JIB scraping: {e}"

# Scrape Banana IT
def search_and_scrape_banana_product(product_name, results):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        search_url = f"https://www.bnn.in.th/th/p?q={product_name.replace(' ', '%20')}&ref=search-result"
        response = requests.get(search_url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            product_list = soup.find('div', {'class': 'product-list'})
            if not product_list:
                results['Banana'] = []

            product_items = product_list.find_all('a', {'class': 'product-link verify product-item'})
            products = []
            for item in product_items:
                product_url = "https://www.bnn.in.th" + item['href']
                product_name_tag = item.find('div', {'class': 'product-name'})
                found_product_name = product_name_tag.text.strip() if product_name_tag else "Product name not found"
                if re.search(product_name.lower(), found_product_name.lower()):  # Check for matching name
                    price_tag = item.find('div', {'class': 'product-price'})
                    product_price = price_tag.text.strip() if price_tag else "Price not found"
                    products.append({"name": found_product_name, "price": product_price, "url": product_url})
            results['Banana'] = filter_products_by_name(products, product_name)
        else:
            results['Banana'] = f"Failed to search Banana IT. Status code: {response.status_code}"
    except Exception as e:
        results['Banana'] = f"Error occurred during Banana IT scraping: {e}"

# ==================== RECOMMENDATION SYSTEM FUNCTIONS ====================

# Global cache variables
recommendation_cache = {} 
impression_history_cache = {}

# Cache expiry times
CACHE_EXPIRY_TIME_SECONDS = 10  # Main recommendation cache expiry (e.g., 10 seconds for hard refresh)
IMPRESSION_HISTORY_TTL_SECONDS = 3600 # Impression history entry TTL (e.g., 1 hour)
IMPRESSION_HISTORY_MAX_ENTRIES = 100 # Max entries per user to prevent memory overflow

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

def normalize_scores(series):
    min_val, max_val = series.min(), series.max()
    if max_val > min_val:
        return (series - min_val) / (max_val - min_val)
    return series

def normalize_engagement(data, user_column='owner_id', engagement_column='PostEngagement'):
    data['NormalizedEngagement'] = data.groupby(user_column)[engagement_column].transform(lambda x: normalize_scores(x))
    return data

def analyze_comments(comments_series):
    comment_counts = []
    for comment_text in comments_series:
        if pd.isna(comment_text) or str(comment_text).strip() == '':
            comment_counts.append(0)
        else:
            individual_comments = [c.strip() for c in str(comment_text).split(';') if c.strip()]
            comment_counts.append(len(individual_comments))
    return comment_counts

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


def split_and_rank_recommendations(recommendations, user_interactions, impression_history, total_posts_in_db):
    unique_recommendations_ids = [int(p) if isinstance(p, float) else p for p in list(dict.fromkeys(recommendations))]
    user_interactions_set = set([int(p) if isinstance(p, float) else p for p in user_interactions])
    impression_history_set = set([int(p) for entry in impression_history for p in [entry['post_id']]]) # Extract IDs from history entries

    final_recommendations_ordered = []

    # 1. Identify "truly unviewed" posts: not interacted with AND not in recent impression history
    truly_unviewed_posts = [
        post_id for post_id in unique_recommendations_ids 
        if post_id not in user_interactions_set and post_id not in impression_history_set
    ]

    # 2. Identify "recently shown but not interacted" posts
    recently_shown_not_interacted = [
        post_id for post_id in unique_recommendations_ids 
        if post_id not in user_interactions_set and post_id in impression_history_set
    ]
    
    # 3. Identify "interacted" posts
    interacted_posts = [
        post_id for post_id in unique_recommendations_ids 
        if post_id in user_interactions_set
    ]

    
    # Calculate target number for "fresh" posts. Let's aim for 25% of the total recommended list length.
    num_fresh_priority = min(len(truly_unviewed_posts), max(10, int(len(unique_recommendations_ids) * 0.25)))

    # Take the freshest posts first
    final_recommendations_ordered.extend(truly_unviewed_posts[:num_fresh_priority])
    remaining_posts_to_mix = [
        post_id for post_id in unique_recommendations_ids 
        if post_id not in set(final_recommendations_ordered)
    ]

    group_A_not_recently_shown = [
        post_id for post_id in remaining_posts_to_mix 
        if post_id not in impression_history_set
    ]
    
    # Group B: Posts that *are* recently shown (recently_shown_not_interacted from above, filtered to remaining_posts_to_mix)
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
    
    # Take the initial fresh batch (e.g., top 30 from truly_unviewed_non_demoted)
    num_unviewed_first_current_run = min(30, len(truly_unviewed_non_demoted))
    final_recommendations_ordered.extend(truly_unviewed_non_demoted[:num_unviewed_first_current_run])
    
    remaining_to_shuffle_and_mix = truly_unviewed_non_demoted[num_unviewed_first_current_run:] + \
                                   remaining_non_demoted_and_not_truly_unviewed + \
                                   demoted_posts
    
    shuffled_segment = []
    # Use a dynamic block size, e.g., 5-10
    block_size = 5 
    
    blocks = [
        remaining_to_shuffle_and_mix[i : i + block_size]
        for i in range(0, len(remaining_to_shuffle_and_mix), block_size)
    ]
    
    for block in blocks:
        block_with_priority = []
        for post_id in block:
            priority_score = 0 # Default: most fresh
            if post_id in posts_to_demote:
                priority_score = 2 # Lowest priority: demoted
            elif post_id in impression_history_set:
                priority_score = 1 # Medium priority: recently shown but not demoted
            block_with_priority.append((post_id, priority_score))
        
        # Sort: lower priority_score first, then random for tie-breaking
        block_with_priority.sort(key=lambda x: (x[1], random.random()))
        
        shuffled_segment.extend([post_id for post_id, _ in block_with_priority])

    final_recommendations_ordered.extend(shuffled_segment)

    print("Top N Unviewed (prioritized):", final_recommendations_ordered[:num_unviewed_first_current_run])
    print("Remaining Overall Recommendations (block-shuffled, with recently shown and demoted pushed back):", final_recommendations_ordered[num_unviewed_first_current_run:])
    print("Final Recommendations (ordered, after mix):", final_recommendations_ordered)

    return final_recommendations_ordered

# ==================== SLIP & PROMPTPAY FUNCTIONS (from Slip.py) ====================


def find_order_by_id(order_id): # ลบ conn=None ออก เพราะตอนนี้ใช้ db.Model.query
    """
    ค้นหา Order ด้วย ID และคืนค่าเป็น Dictionary พร้อมข้อมูลที่จำเป็น
    เพิ่ม renew_ads_id, package_id, และ show_at เพื่อรองรับการต่ออายุและโฆษณาใหม่
    """
    order = Order.query.filter_by(id=order_id).first()
    if not order:
        return None
    return {
        'id': order.id,
        'user_id': order.user_id,
        'amount': order.amount,
        'status': order.status,
        'promptpay_qr_payload': order.promptpay_qr_payload,
        'slip_image': order.slip_image, # ดึง slip_image มาด้วย
        'renew_ads_id': order.renew_ads_id, # เพิ่ม
        'package_id': order.package_id,     # เพิ่ม
        'show_at': order.show_at            # เพิ่ม
    }


def find_ad_by_order_id(order_id): # ลบ conn=None ออก
    """
    ค้นหา Ad ที่ผูกกับ Order ID และคืนค่าเป็น Dictionary
    เพิ่ม show_at เพื่อรองรับ Ad ใหม่
    """
    ad = Ad.query.filter_by(order_id=order_id).first()
    if not ad:
        return None
    return {
        'id': ad.id,
        'status': ad.status,
        'expiration_date': ad.expiration_date,
        'show_at': ad.show_at # เพิ่ม
    }


def find_ad_by_id(ad_id): # ฟังก์ชันใหม่สำหรับค้นหา Ad ด้วย Ad ID
    """
    ค้นหา Ad ด้วย Ad ID และคืนค่าเป็น Dictionary พร้อมข้อมูลที่จำเป็น
    ใช้สำหรับดึงข้อมูล Ad เดิมที่ต้องการต่ออายุ
    """
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


def get_ad_package_duration(package_id): # ฟังก์ชันใหม่สำหรับดึง duration_days
    """
    ดึง duration_days จาก AdPackage ด้วย package_id
    """
    pkg = AdPackage.query.filter_by(package_id=package_id).first()
    if not pkg:
        print(f"❌ [ERROR] AdPackage with ID {package_id} not found.")
        return None
    return pkg.duration_days


def update_status_and_slip_info(order_id, new_status, slip_image_path, slip_transaction_id):
    """
    อัปเดตสถานะ Order และบันทึกข้อมูลสลิป
    """
    order = Order.query.filter_by(id=order_id).first()
    if not order:
        print(f"❌ [ERROR] Order ID {order_id} not found for status update.")
        return False
    order.status = new_status
    order.slip_image = slip_image_path
    # --- ลบบรรทัดนี้ออก ---
    # order.slip_transaction_id = slip_transaction_id # เพิ่มการบันทึก transaction ID
    order.updated_at = datetime.now()
    try:
        db.session.commit()
        print(f"✅ Order ID: {order_id} status updated to '{new_status}' with slip info.")
        return True
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error updating order status for ID {order_id}: {e}")
        return False


def update_ad_status(ad_id, new_status): # ลบ conn=None ออก
    """
    อัปเดตสถานะของ Ad เท่านั้น (สำหรับ Ad ใหม่ หรือ Ad ต่ออายุที่ต้องการเปลี่ยนสถานะเฉยๆ)
    ไม่กระทบ expiration_date หรือ show_at
    """
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


def update_ad_for_renewal(ad_id, new_status, new_expiration_date): # ฟังก์ชันใหม่สำหรับอัปเดต Ad ที่ต่ออายุ
    """
    อัปเดตสถานะและวันหมดอายุของ Ad สำหรับการต่ออายุ
    *ไม่เปลี่ยนแปลง show_at เดิมของ Ad*
    """
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


def update_order_with_promptpay_payload_db(order_id, payload_to_store_in_db): # ลบ conn=None ออก
    """
    บันทึก PromptPay QR Payload ลงใน Order
    """
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


def create_advertisement_db(order_data): # ลบ conn=None ออก
    """
    สร้าง Ad ใหม่สำหรับ Order ที่ชำระเงินแล้ว (ในกรณีที่เป็นโฆษณาใหม่)
    กำหนด status เริ่มต้นเป็น 'paid' และใช้ show_at จาก order_data
    """
    now = datetime.now()
    default_title = f"Advertisement for Order {order_data['id']}"
    default_content = "This is a new advertisement pending admin approval after payment."
    
    # show_at ของ Ad ใหม่จะใช้ค่าจาก order_data['show_at'] ถ้ามี
    # ซึ่งโดยปกติแล้ว order_data['show_at'] จะมาจากตอนสร้าง Order
    ad_show_at = order_data.get('show_at', now) # กำหนดค่าเริ่มต้นเป็น datetime.now() หากไม่มี show_at ใน order_data

    ad = Ad(
        user_id=order_data['user_id'],
        order_id=order_data['id'],
        title=default_title,
        content=default_content,
        link="",
        image="",
        status='paid', # เริ่มต้นเป็น paid เพราะสลิปผ่านแล้ว
        created_at=now,
        updated_at=now,
        show_at=ad_show_at # กำหนด show_at
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


def generate_promptpay_qr_for_order(order_id):
    """
    สร้าง PromptPay QR Code สำหรับ Order ที่กำหนด
    เงื่อนไข:
    - Order ต้องมี status เป็น 'approved' (สำหรับโฆษณาใหม่)
    - หรือ status เป็น 'pending' และ renew_ads_id ไม่เป็น null (สำหรับโฆษณาต่ออายุ)
    """
    order = find_order_by_id(order_id)
    if not order:
        print(f"❌ [WARN] Order ID {order_id} not found for QR generation.")
        return {"success": False, "message": "ไม่พบคำสั่งซื้อ"}

    # เงื่อนไขการ Generate QR Code ตามที่ตกลงกัน
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
    if hasattr(img, 'get_image'): # Pillow specific
        img.get_image().save(buffered, "PNG")
    else: # qrcode library's default image object
        img.save(buffered, "PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return {"success": True, "message": "สร้าง QR Code สำเร็จ", "qrcode_base64": img_b64, "payload": original_scannable_payload}


def can_upload_slip(order):
    """
    คืนค่า True ถ้า order สามารถอัปโหลด slip ได้ตามเงื่อนไข:
    - status เป็น 'approved' (สำหรับโฆษณาใหม่)
    - หรือ status เป็น 'pending' และ renew_ads_id ไม่เป็น None (สำหรับโฆษณาต่ออายุ)
    """
    if not order:
        return False
    
    is_new_ad_approved = order["status"] == 'approved' and order.get("renew_ads_id") is None
    is_renewal_ad_pending = order["status"] == 'pending' and order.get("renew_ads_id") is not None
    
    return is_new_ad_approved or is_renewal_ad_pending


def format_thai_date(date_obj):
    """
    ฟังก์ชันสำหรับจัดรูปแบบวันที่ให้เป็นภาษาไทยและปีพุทธศักราช
    รับค่า date_obj: สามารถเป็น datetime.date, datetime.datetime, หรือ string
    """
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


def notify_ads_status_change(db, ad_id: int, new_status: str, admin_notes: str = None, duration_days_from_renewal: int = None) -> bool:
    """
    สร้างและบันทึก notification เมื่อสถานะโฆษณาเปลี่ยนแปลง หรือเมื่อมีการต่ออายุโฆษณาสำเร็จ.
    ใช้ SQLAlchemy db.session ในการ query ข้อมูลและ COMMIT การแจ้งเตือนทันที.

    Parameters:
    - db: instance ของ SQLAlchemy object ที่เชื่อมต่อกับฐานข้อมูลแล้ว
    - ad_id (int): รหัสของโฆษณาที่สถานะมีการเปลี่ยนแปลง
    - new_status (str): สถานะใหม่ของโฆษณา
    - admin_notes (str, optional): ข้อความบันทึกจากแอดมิน (ใช้ในกรณี 'rejected')
    - duration_days_from_renewal (int, optional): จำนวนวันที่ต่ออายุ (ใช้ในกรณี 'active' จากการต่ออายุ)
    """
    try:
        # ขั้นตอนที่ 1: ดึง user_id และ expiration_date ล่าสุดจากตาราง ads
        ads_query = text('SELECT user_id, expiration_date FROM ads WHERE id = :ad_id')
        ads_result = db.session.execute(ads_query, {'ad_id': ad_id}).fetchone()

        if not ads_result:
            print(f"❌ [WARN] notify_ads_status_change: Ad ID {ad_id} not found in 'ads' table.")
            return False
        
        user_id = ads_result[0] # user_id
        expiration_date = ads_result[1] # expiration_date

        content = ''
        
        # เป็ดน้อยจะเปลี่ยน logic ในการดึง duration_from_package_db
        # ให้ดึงเฉพาะเมื่อ duration_days_from_renewal ไม่มีค่า (คือไม่ได้ส่งมาจาก verify_payment_and_update_status)
        duration_from_package_db = None
        if duration_days_from_renewal is None: # ถ้าไม่มี duration_days ส่งมาโดยตรง ให้พยายามดึงจาก DB
            order_package_query = text("""
                SELECT ap.duration_days, o.status as order_status
                FROM orders o
                JOIN ad_packages ap ON o.package_id = ap.package_id
                WHERE o.renew_ads_id = :ad_id -- แก้ไขตรงนี้: ลบ OR o.ads_id = :ad_id ออกไป
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


        # กำหนดข้อความแจ้งเตือนตามเงื่อนไข
        if new_status == 'active':
            # ใช้ duration_days_from_renewal ก่อน ถ้ามีค่า (ซึ่งจะเป็นค่าที่ถูกต้องจากการคำนวณใน verify_payment_and_update_status)
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
        
        # ขั้นตอนสุดท้าย: บันทึกข้อมูลการแจ้งเตือนลงในตาราง notifications
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


def verify_payment_and_update_status(order_id, slip_image_path, payload_from_client, db):
    """
    ฟังก์ชันหลักในการตรวจสอบสลิปการโอนเงินและอัปเดตสถานะคำสั่งซื้อ/โฆษณา.
    จะรองรับทั้งการชำระเงินสำหรับโฆษณาใหม่และการต่ออายุโฆษณา.
    """
    print(f"\n--- Processing payment for Order ID: {order_id} ---")
    print(f"Slip image path: {slip_image_path}")
    print(f"Payload (from client - original QR data): {payload_from_client}")

    # ดึงข้อมูล Order มาก่อน
    order = find_order_by_id(order_id)
    if not order:
        print(f"❌ [ERROR] Order ID {order_id} not found.")
        return {"success": False, "message": "ไม่พบคำสั่งซื้อ"}

    try:
        # ตรวจสอบเงื่อนไขการอัปโหลดสลิป
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

        # --- ส่วนตรวจสอบ Ad ที่เกี่ยวข้องก่อนเรียก SlipOK API ---
        ad_related = None
        if order.get("renew_ads_id") is not None:
            # กรณีต่ออายุโฆษณา: ตรวจสอบ Ad เดิม
            ad_related = find_ad_by_id(order["renew_ads_id"])
            if not ad_related:
                print(f"❌ [ERROR] Associated ad for renewal (ID {order['renew_ads_id']}) not found for Order ID {order_id}.")
                return {"success": False, "message": "ไม่พบโฆษณาที่ต้องการต่ออายุ"}

            # ตรวจสอบว่าโฆษณาเดิมหมดอายุแล้วหรือไม่
            today = datetime.now().date()
            if isinstance(ad_related.get('expiration_date'), date) and ad_related['expiration_date'] < today:
                print(f"❌ [WARN] Cannot renew ad ID {ad_related['id']} for Order ID {order_id}. Ad has expired on {ad_related['expiration_date'].strftime('%Y-%m-%d')}.")
                return {"success": False, "message": "ไม่สามารถต่ออายุโฆษณาได้ เนื่องจากโฆษณาหมดอายุแล้ว"}

            # ตรวจสอบสถานะ Ad เดิมที่สามารถต่ออายุได้ (active, expiring_soon, paused)
            if ad_related.get('status') not in ['active', 'expiring_soon', 'paused']:
                print(f"❌ [WARN] Associated ad ID {ad_related['id']} for Order ID {order_id} is not in a renewable status. Current ad status: {ad_related['status']}.")
                return {"success": False, "message": "โฆษณาสำหรับคำสั่งซื้อนี้ไม่อยู่ในสถานะที่สามารถต่ออายุได้"}
        else:
            # กรณีโฆษณาใหม่: ตรวจสอบ Ad ที่ผูกกับ Order ID (ถ้ามีและสถานะไม่ถูกต้อง)
            ad_related = find_ad_by_order_id(order_id)
            if ad_related and ad_related.get('status') in ['active', 'rejected', 'paid']:
                print(f"❌ [WARN] Associated ad for Order ID {order_id} is already processed. Current ad status: {ad_related['status']}.")
                return {"success": False, "message": "โฆษณาสำหรับคำสั่งซื้อนี้มีการดำเนินการไปแล้ว"}

        # --- เรียก SlipOK API จริง ---
        SLIP_OK_API_ENDPOINT = os.getenv("SLIP_OK_API_ENDPOINT", "https://api.slipok.com/api/line/apikey/49130")
        SLIP_OK_API_KEY = os.getenv("SLIP_OK_API_KEY", "SLIPOKKBE52WN")

        if not os.path.exists(slip_image_path):
            print(f"❌ [ERROR] Slip image file not found at '{slip_image_path}'")
            return {"success": False, "message": "ไม่พบไฟล์รูปภาพสลิป"}

        response = None # กำหนดค่าเริ่มต้นให้ response
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
            response.raise_for_status() # จะ raise HTTPError ถ้าสถานะไม่ใช่ 2xx

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

        # ตรวจสอบยอดเงิน
        if abs(slip_amount - float(order.get("amount"))) > 0.01:
            print(f"❌ [WARN] Amount mismatch. Order: {order.get('amount')}, Slip: {slip_amount}")
            return {"success": False, "message": f"ยอดเงินไม่ถูกต้อง (ต้องการ {order.get('amount'):.2f} บาท แต่ได้รับ {slip_amount:.2f} บาท)"}

        # --- เริ่มต้น Transaction เพื่ออัปเดตฐานข้อมูล ---
        # ใช้ db.session ของ SQLAlchemy
        if not update_status_and_slip_info(order_id, "paid", slip_image_path, slip_transaction_id_from_api):
            raise Exception("Failed to update order status and slip info.")

        ad_id_to_return = None
        ad_new_status_for_notification = None # เพิ่มตัวแปรเพื่อเก็บสถานะสำหรับแจ้งเตือน

        # ตรวจสอบว่าเป็น Order สำหรับการต่ออายุโฆษณาหรือไม่
        if order.get("renew_ads_id") is not None:
            # กรณีต่ออายุโฆษณา: อัปเดต Ad เดิม
            current_ad = find_ad_by_id(order["renew_ads_id"])
            if not current_ad:
                raise Exception(f"Ad with ID {order['renew_ads_id']} not found for renewal processing after order update.")

            duration_days = get_ad_package_duration(order["package_id"])
            if duration_days is None:
                raise Exception(f"Ad package duration not found for package_id {order['package_id']} for renewal.")

            # --- Logic การคำนวณวันหมดอายุใหม่ ---
            original_expiration = current_ad.get('expiration_date')

            renewal_start_date_candidate = None

            if original_expiration:
                if isinstance(original_expiration, date) and not isinstance(original_expiration, datetime):
                    renewal_start_date_candidate = datetime.combine(original_expiration, datetime.min.time())
                elif isinstance(original_expiration, datetime):
                    renewal_start_date_candidate = original_expiration

            if original_expiration: # กรณีต่ออายุโฆษณาที่มีวันหมดอายุเดิม
                if renewal_start_date_candidate:
                    calculated_renewal_start = renewal_start_date_candidate + timedelta(days=1)
                else: # Fallback ถ้า renewal_start_date_candidate เป็น None
                    calculated_renewal_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

                if calculated_renewal_start.date() < datetime.now().date(): # ถ้าวันที่เริ่มต้นคำนวณย้อนหลังไปแล้ว
                    actual_renewal_start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                else:
                    actual_renewal_start_date = calculated_renewal_start

                new_expiration_date = actual_renewal_start_date + timedelta(days=duration_days - 1)

            else: # กรณีโฆษณาใหม่ (ไม่ควรเข้าบล็อกนี้หาก renew_ads_id ไม่ใช่ None)
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

            # อัปเดตโฆษณาด้วยสถานะ 'active' และวันหมดอายุใหม่
            if not update_ad_for_renewal(current_ad['id'], "active", new_expiration_date.date()):
                raise Exception("Failed to update existing ad status and expiration date for renewal.")

            ad_id_to_return = current_ad['id']
            ad_new_status_for_notification = 'active' # สถานะสำหรับแจ้งเตือนการต่ออายุ

        else:
            # กรณีโฆษณาใหม่: สร้างหรืออัปเดต Ad
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
            ad_new_status_for_notification = 'paid' # สถานะสำหรับแจ้งเตือนโฆษณาใหม่ที่ชำระแล้ว

        db.session.commit() # *** COMMIT TRANSACTION ที่นี่ สำหรับ Order และ Ad Status ***
        print(f"✅ [INFO] Transaction committed successfully for Order ID: {order_id}.")

        # *** นี่คือจุดที่จะเรียกฟังก์ชันแจ้งเตือน หลังจาก Transaction สำเร็จ ***
        # notify_ads_status_change จะทำการ commit ตัวเอง
        if ad_id_to_return and ad_new_status_for_notification:
            notify_ads_status_change(db, ad_id_to_return, ad_new_status_for_notification)

        # กำหนดข้อความตอบกลับตามประเภทโฆษณา
        if order.get("renew_ads_id") is not None:
             # ดึง duration_days อีกครั้งเพื่อส่งข้อความที่ถูกต้อง
            duration_days = get_ad_package_duration(order["package_id"])
            message = f"ชำระเงินสำเร็จ! โฆษณาของคุณได้รับการต่ออายุเพิ่มอีก {duration_days} วันเรียบร้อยแล้ว"
        else:
            message = "ชำระเงินสำเร็จ! กรุณารอแอดมินตรวจสอบ"

        return {"success": True, "message": message, "ad_id": ad_id_to_return}

    except requests.exceptions.Timeout:
        # หากเกิด Timeout จากการเรียก SlipOK API
        try:
            db.session.rollback() # Rollback หากเกิดข้อผิดพลาดก่อน commit
        except Exception as rollback_e:
            print(f"⚠️ [WARN] Error during rollback: {rollback_e}")
        print(f"❌ [API ERROR] SlipOK Timeout: API ไม่ตอบกลับภายในเวลาที่กำหนดสำหรับ Order ID: {order_id}.")
        return {"success": False, "message": "ระบบตรวจสอบสลิปตอบกลับช้าเกินไป โปรดลองอีกครั้ง"}
    except requests.exceptions.HTTPError as e:
        # หากเกิด HTTP Error (สถานะ 4xx, 5xx)
        try:
            db.session.rollback() # Rollback หากเกิดข้อผิดพลาดก่อน commit
        except Exception as rollback_e:
            print(f"⚠️ [WARN] Error during rollback: {rollback_e}")

        # เพิ่มการดึงข้อความจาก SlipOK response มาแสดงใน log ให้ชัดเจน
        slipok_error_message = "ไม่ทราบข้อผิดพลาดจาก SlipOK API"
        if response is not None: # ตรวจสอบว่า response มีค่า (คือเรียก API ไปแล้ว)
            try:
                error_details = response.json()
                slipok_error_message = error_details.get('message', 'ไม่พบข้อความผิดพลาดจาก SlipOK')
                print(f"❌ [API ERROR] HTTP Error {e.response.status_code} for Order ID: {order_id}. SlipOK Message: {slipok_error_message}. URL: {e.request.url}")
                print(f"    Full SlipOK Response Body: {response.text}") # เพิ่มบรรทัดนี้เพื่อ debug เพิ่มเติม
            except Exception as json_e:
                # กรณีที่ response.json() ล้มเหลว (อาจจะไม่ใช่ JSON)
                print(f"❌ [API ERROR] HTTP Error {e.response.status_code} for Order ID: {order_id}. URL: {e.request.url}. Cannot parse SlipOK response as JSON. Error: {json_e}")
                print(f"    Full SlipOK Response Text (Non-JSON): {response.text}") # พิมพ์ text ออกมาตรงๆ
                slipok_error_message = f"เกิดข้อผิดพลาดในการประมวลผลคำตอบจาก SlipOK: {response.text[:100]}..." # แสดงส่วนแรกของ text
        else:
            # กรณีที่ response เป็น None (ไม่น่าจะเกิดขึ้นหลัง raise_for_status แต่ก็เผื่อไว้)
            print(f"❌ [API ERROR] HTTP Error for Order ID: {order_id}. Error: {e}. (No SlipOK response object found)")

        return {"success": False, "message": f"การตรวจสอบสลิปไม่สำเร็จ: {slipok_error_message}"}

    except requests.exceptions.RequestException as e:
        # หากเกิดข้อผิดพลาดอื่นๆ ที่เกี่ยวกับ requests (เช่น ConnectionError, DNS issues)
        try:
            db.session.rollback() # Rollback หากเกิดข้อผิดพลาดก่อน commit
        except Exception as rollback_e:
            print(f"⚠️ [WARN] Error during rollback: {rollback_e}")
        print(f"❌ [API ERROR] Connection Error: ไม่สามารถเชื่อมต่อกับ SlipOK API สำหรับ Order ID: {order_id}. Error: {e}")
        return {"success": False, "message": f"เกิดข้อผิดพลาดในการเชื่อมต่อกับระบบตรวจสอบสลิป: {e}"}
    except ValueError:
        # หากแปลงยอดเงินไม่ได้
        try:
            db.session.rollback() # Rollback หากเกิดข้อผิดพลาดก่อน commit
        except Exception as rollback_e:
            print(f"⚠️ [WARN] Error during rollback: {rollback_e}")
        print(f"❌ [API ERROR] Data Parsing Error: ไม่สามารถอ่านยอดเงินจาก SlipOK response ได้สำหรับ Order ID: {order_id}.")
        return {"success": False, "message": "รูปแบบยอดเงินจากระบบตรวจสอบสลิปไม่ถูกต้อง"}
    except Exception as e:
        # หากมีข้อผิดพลาดอื่นๆ ที่ไม่ได้ระบุไว้ใน try-except block ข้างต้น
        try:
            # ใช้ db.session.rollback() เมื่อเกิด exception ทั่วไป
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

        # ตรวจสอบ user_id (mockup: ต้องมี user_id)
        if not user_id:
            return jsonify({"error": "You are not authorized to create a post for this user"}), 403

        # รับ URL ของรูปภาพที่อัปโหลด
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
                    # ลบไฟล์ภาพที่ไม่เหมาะสม
                    if os.path.exists(photo_path):
                        os.remove(photo_path)
                    
                    # เก็บข้อมูลภาพที่ไม่เหมาะสม
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
                # หากเกิดข้อผิดพลาด ให้ลบไฟล์และแจ้งเตือน
                if 'photo_path' in locals() and os.path.exists(photo_path):
                    os.remove(photo_path)
                invalid_photos.append({
                    "filename": photo.filename,
                    "reason": "ไม่สามารถประมวลผลภาพได้",
                    "details": {"error": str(e)}
                })
        
        # หากมีภาพที่ไม่เหมาะสม ให้แจ้งเตือนและไม่สร้าง post
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

        # รับ URL ของวิดีโอที่อัปโหลด (mockup)
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

        # บันทึกข้อมูลลงฐานข้อมูล MySQL
        try:
            # สร้าง SQL query สำหรับเพิ่มโพสต์ใหม่
            insert_query = text("""
                INSERT INTO posts (user_id, Title, content, ProductName, CategoryID, photo_url, video_url, status, updated_at)
                VALUES (:user_id, :title, :content, :product_name, :category_id, :photo_urls, :video_urls, 'active', NOW())
            """)
            
            # Execute query
            result = db.session.execute(insert_query, {
                'user_id': user_id,
                'title': title,
                'content': content,
                'product_name': product_name,
                'category_id': category,  # ต้องส่งค่าตัวแปร category ไปที่ CategoryID
                'photo_urls': photo_urls_json,
                'video_urls': video_urls_json
            })
            
            # Commit การเปลี่ยนแปลง
            db.session.commit()
            
            # ดึง post_id ที่เพิ่งสร้าง
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
        
        # อ่านค่า existing_photos และ existing_videos จาก JSON string ถ้ามี
        # เนื่องจาก form.getlist จะคืนค่าเป็น list ของ string แม้ว่าจะเป็น JSON string ก็ตาม
        # เราต้องแปลงมันเป็น Python list ที่ถูกต้อง
        existing_photos_str = request.form.get('existing_photos')
        existing_videos_str = request.form.get('existing_videos')
        
        # พยายามโหลด JSON, ถ้าไม่เป็น JSON หรือว่างเปล่า ก็ให้เป็น list ว่าง
        existing_photos = json.loads(existing_photos_str) if existing_photos_str else []
        existing_videos = json.loads(existing_videos_str) if existing_videos_str else []
        
        photos = request.files.getlist('photo')
        videos = request.files.getlist('video')

        if not user_id:
            return jsonify({"error": "You are not authorized to update this post"}), 403

        # รวมรูปภาพเดิมและรูปภาพใหม่ที่อัปโหลด
        photo_urls = existing_photos if existing_photos else []
        invalid_photos = []

        for photo in photos:
            # กำหนด photo_path ไว้ก่อน
            photo_path = None
            if not photo or not photo.filename:
                continue
                
            try:
                filename = secure_filename(photo.filename)
                photo_path = os.path.join(UPLOAD_FOLDER, filename)
                photo.save(photo_path) # บันทึกไฟล์ก่อนตรวจสอบ

                print(f"Processing new photo for update: {filename}")
                is_nude, result = nude_predict_image(photo_path)
                
                if is_nude:
                    print(f"NSFW detected in {filename} during update")
                    # ลบไฟล์ภาพที่ไม่เหมาะสมทันที
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
                    # ไม่ต้องลบไฟล์ที่นี่ เพราะจะถูกเก็บไว้เป็นส่วนหนึ่งของโพสต์
                    
            except Exception as e:
                print(f"Error processing photo {photo.filename} during update: {e}")
                # หากเกิดข้อผิดพลาด ให้ลบไฟล์และแจ้งเตือน
                if photo_path and os.path.exists(photo_path): # ตรวจสอบ photo_path ก่อนใช้
                    os.remove(photo_path)
                invalid_photos.append({
                    "filename": photo.filename,
                    "reason": "ไม่สามารถประมวลผลภาพได้",
                    "details": {"error": str(e)}
                })
        
        # หากมีภาพที่ไม่เหมาะสม ให้แจ้งเตือนและไม่อัปเดต post
        if invalid_photos:
            print("แจ้งเตือน user: พบภาพไม่เหมาะสมในระหว่างการอัปเดต")
            # ถ้ามี invalid_photos จะไม่ทำการอัปเดตข้อมูลใน DB เลย
            return jsonify({
                "status": "warning",
                "message": "กรุณาเปลี่ยนภาพแล้วลองใหม่อีกครั้ง",
                "invalid_photos": invalid_photos,
                "valid_photos": photo_urls
            }), 400

        # รวมวิดีโอเดิมและวิดีโอใหม่ที่อัปโหลด
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

        # อัปเดตข้อมูลในฐานข้อมูล MySQL
        try:
            # สร้าง SQL query สำหรับอัปเดตโพสต์
            update_query = text("""
                UPDATE posts 
                SET Title = :title, content = :content, ProductName = :product_name, 
                    CategoryID = :category_id, photo_url = :photo_urls, video_url = :video_urls, 
                    updated_at = NOW()
                WHERE id = :post_id AND user_id = :user_id
            """)
            
            # Execute query
            result = db.session.execute(update_query, {
                'post_id': id,
                'user_id': user_id,
                'title': Title,
                'content': content,
                'product_name': ProductName,
                'category_id': CategoryID, # แก้เป็น CategoryID ให้ตรงกับชื่อคอลัมน์ใน DB
                'photo_urls': photo_urls_json,
                'video_urls': video_urls_json
            })
            
            # Commit การเปลี่ยนแปลง
            db.session.commit()
            
            # ตรวจสอบว่ามีแถวที่ถูกอัปเดตหรือไม่
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

        # --- ส่วนจัดการและตรวจสอบวันเกิด ---
        # เนื่องจาก Front-end จะส่งมาเป็น YYYY-MM-DD แล้ว
        # เรายังคงให้ Back-end ยืดหยุ่นในการ parse รูปแบบอื่น ๆ เผื่อไว้
        birthday = None
        date_formats = [
            "%Y-%m-%d",    # รูปแบบหลักจาก Front-end
            "%d/%m/%Y",    # เผื่อไว้ถ้ามีคนใช้รูปแบบเก่า
            "%m/%d/%Y",
            "%Y/%m/%d",
            "%d-%m-%Y",
            # ถ้า Front-end อาจส่งเป็น d MMM yyyy (เช่น "28 ก.ค. 2568" หรือ "28 Jul 2025")
            # ถ้า Locale ของ Back-end ไม่ใช่ไทย อาจต้องระบุ locale ใน strptime
            # "%d %b %Y",  # For "28 Jul 2025" (English short month)
            # "%d %B %Y",  # For "28 July 2025" (English full month)
            # ถ้าเป็นภาษาไทย อาจต้องใช้ไลบรารีอื่นช่วย parse หรือกำหนด Locale
        ]
        
        # เพิ่มการลอง parse แบบ ISO standard ที่ไม่ขึ้นกับ locale
        try:
            birthday = datetime.fromisoformat(birthday_str).strftime("%Y-%m-%d")
        except ValueError:
            # ถ้า fromisoformat ไม่ได้ผล ให้ลองวิธีอื่น
            for fmt in date_formats:
                try:
                    birthday = datetime.strptime(birthday_str, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue

        if birthday is None:
            return jsonify({"error": "รูปแบบวันเกิดไม่ถูกต้อง โปรดระบุในรูปแบบ YYYY-MM-DD"}), 400
        # --- จบส่วนจัดการวันเกิด ---

        # --- ส่วนจัดการและตรวจสอบรูปภาพโปรไฟล์ด้วย AI ---
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
        # --- จบส่วนจัดการรูปภาพโปรไฟล์ ---

        check_username_query = text("""
            SELECT id FROM users WHERE username = :username AND id != :user_id
        """)
        check_results = db.session.execute(check_username_query, {
            'username': username,
            'user_id': userId
        }).fetchall()

        if len(check_results) > 0:
            return jsonify({"error": "ชื่อผู้ใช้นี้มีคนใช้แล้วงับ"}), 400

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

        if profile_image_path:
            update_profile_query += ", picture = :picture"
            update_data['picture'] = profile_image_path

        update_profile_query += " WHERE id = :user_id"

        try:
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


# Web Scraping Route
@app.route('/ai/search', methods=['GET'])
def search_product():
    product_name = request.args.get('productname')
    if not product_name:
        return jsonify({"error": "Please provide a product name"}), 400

    results = {product_name: {}}

    # สร้าง thread สำหรับการดึงข้อมูลจากแต่ละร้าน
    threads = []
    threads.append(threading.Thread(target=search_and_scrape_advice_product, args=(product_name, results[product_name])))
    threads.append(threading.Thread(target=search_and_scrape_jib_product_from_search, args=(product_name, results[product_name])))
    threads.append(threading.Thread(target=search_and_scrape_banana_product, args=(product_name, results[product_name])))

    # รัน threads
    for thread in threads:
        thread.start()

    # รอให้ทุก thread ทำงานเสร็จ
    for thread in threads:
        thread.join()

    return jsonify(results)

# Recommendation Route
@app.route('/ai/recommend', methods=['POST'])
@verify_token
def recommend():
    try:
        user_id = request.user_id
        now = datetime.now()

        if user_id in recommendation_cache:
            cached_data, cache_timestamp = recommendation_cache[user_id]
            if (now - cache_timestamp).total_seconds() < (CACHE_EXPIRY_TIME_SECONDS / 2):
                print(f"Returning VERY FRESH cached recommendations for user_id: {user_id}")
                return jsonify(cached_data)
            print(f"Cached recommendations for user_id: {user_id} are still valid but slightly old. Recalculating for freshness.")

        content_based_data, collaborative_data = load_data_from_db()

        try:
            knn = joblib.load('KNN_Model.pkl')
            collaborative_model = joblib.load('Collaborative_Model.pkl')
            tfidf = joblib.load('TFIDF_Model.pkl')
            
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

        categories = [
            'Electronics_Gadgets', 'Furniture', 'Outdoor_Gear', 'Beauty_Products', 'Accessories'
        ]

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

        user_interactions = collaborative_data[collaborative_data['user_id'] == user_id]['post_id'].tolist()
        
        current_impression_history = impression_history_cache.get(user_id, [])
        print(f"Current Impression History for user {user_id}: {[entry['post_id'] for entry in current_impression_history]}")

        # แก้ไขตรงนี้งับ: เพิ่ม total_posts_in_db เข้าไป
        final_recommendations_ids = split_and_rank_recommendations(
            recommendations, 
            user_interactions, 
            current_impression_history,
            len(content_based_data) # ส่งจำนวนโพสต์ทั้งหมดใน content_based_data เข้าไป
        )
        
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

        recommendation_cache[user_id] = (output, now)

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
    # เรียกใช้ฟังก์ชัน generate_promptpay_qr_for_order ที่ได้รับการปรับแก้แล้ว
    result = generate_promptpay_qr_for_order(order_id)
    if not result['success']:
        # หากไม่สำเร็จ จะส่งข้อความผิดพลาดและสถานะ HTTP 400
        return jsonify(result), 400
    
    # หากสำเร็จ จะส่งข้อมูล QR Code และ payload กลับไป
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
        order_status = order.get('status', 'N/A') # ใช้ .get() เพื่อความปลอดภัย
        renew_ad_id = order.get('renew_ads_id', 'N/A') # ใช้ .get() เพื่อความปลอดภัย
        print(f"❌ [WARN] API Verify Slip: Order ID {order_id} not eligible for slip upload. Current status: {order_status}. Renew Ad: {renew_ad_id}.")
        return jsonify({'success': False, 'message': 'ไม่สามารถอัปโหลดสลิปได้ เนื่องจากสถานะคำสั่งซื้อไม่ถูกต้อง หรือยังไม่ได้รับการอนุมัติ'}), 400

    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    slip_dir = 'Slip'
    if not os.path.exists(slip_dir):
        os.makedirs(slip_dir)
    save_path = os.path.join(slip_dir, unique_filename)
    file.save(save_path)
    
    print(f"✅ [INFO] API Verify Slip: Slip image uploaded to {save_path} for Order ID {order_id}.")
    print(f"✅ [INFO] API Verify Slip: Payload from client (QR Code data): {payload}.")

    # --- แก้ไขตรงนี้: สลับตำแหน่ง argument กลับให้ถูกต้องและเพิ่ม db object ---
    # verify_payment_and_update_status คาดหวัง (order_id, slip_image_path, client_payload, db)
    result = verify_payment_and_update_status(order_id, save_path, payload, db)

    if not result.get('success'):
        return jsonify(result), 400

    return jsonify(result), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)