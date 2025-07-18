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

from datetime import datetime, timezone
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

# ==================== SLIP & PROMPTPAY FUNCTIONS (from Slip.py) ====================
import qrcode
from qrcode.constants import ERROR_CORRECT_H
import uuid
import base64
import io
try:
    from promptpay import qrcode as promptpay_qrcode
except ImportError:
    promptpay_qrcode = None  # จะ refactor ให้รองรับกรณีไม่มี promptpay ทีหลัง

# TODO: Refactor ฟังก์ชันด้านล่างให้ใช้ SQLAlchemy/db.session แทน mysql.connector
# TODO: Refactor config เช่น SLIP_OK_API_KEY, PROMPTPAY_RECEIVER_ID ให้ใช้ app.config หรือ os.getenv ตาม AppAI.py

# --- ฟังก์ชัน DB Interactions (จะ refactor ให้ใช้ SQLAlchemy) ---
def find_order_by_id(order_id, conn=None):
    order = Order.query.filter_by(id=order_id).first()
    if not order:
        return None
    return {
        'id': order.id,
        'user_id': order.user_id,
        'amount': order.amount,
        'order_status': order.order_status,
        'promptpay_qr_payload': order.promptpay_qr_payload
    }

def find_ad_by_order_id(order_id, conn=None):
    ad = Ad.query.filter_by(order_id=order_id).first()
    if not ad:
        return None
    return {
        'id': ad.id,
        'status': ad.status,
        'expiration_date': ad.expiration_date
    }

def update_order_status_and_slip_info(order_id, new_status, slip_image_path, slip_transaction_id, conn=None):
    order = Order.query.filter_by(id=order_id).first()
    if not order:
        return False
    order.order_status = new_status
    order.slip_image = slip_image_path
    order.updated_at = datetime.now()
    try:
        db.session.commit()
        print(f"✅ Order ID: {order_id} status updated to '{new_status}' with slip info.")
        return True
    except Exception as e:
        db.session.rollback()
        print(f"Error updating order status for ID {order_id}: {e}")
        return False

def update_ad_status(ad_id, new_status, conn=None):
    ad = Ad.query.filter_by(id=ad_id).first()
    if not ad:
        return False
    ad.status = new_status
    ad.updated_at = datetime.now()
    try:
        db.session.commit()
        print(f"✅ Ad ID: {ad_id} status updated to '{new_status}'.")
        return True
    except Exception as e:
        db.session.rollback()
        print(f"Error updating ad status for ID {ad_id}: {e}")
        return False

def update_order_with_promptpay_payload_db(order_id, payload_to_store_in_db, conn=None):
    order = Order.query.filter_by(id=order_id).first()
    if not order:
        return False
    order.promptpay_qr_payload = payload_to_store_in_db
    order.updated_at = datetime.now()
    try:
        db.session.commit()
        print(f"✅ Order ID: {order_id} updated with PromptPay payload.")
        return True
    except Exception as e:
        db.session.rollback()
        print(f"Error updating order with PromptPay payload: {e}")
        return False

def create_advertisement_db(order_data, conn=None):
    now = datetime.now()
    default_title = f"Advertisement for Order {order_data['id']}"
    default_content = "This is a new advertisement pending admin approval after payment."
    ad = Ad(
        user_id=order_data['user_id'],
        order_id=order_data['id'],
        title=default_title,
        content=default_content,
        link="",
        image="",
        status='paid',
        created_at=now,
        updated_at=now
    )
    try:
        db.session.add(ad)
        db.session.commit()
        print(f"🚀 Advertisement ID: {ad.id} created for Order ID: {order_data['id']} with status 'paid'.")
        return ad.id
    except Exception as e:
        db.session.rollback()
        print(f"Error creating advertisement: {e}")
        return None

# --- ฟังก์ชันสร้าง PromptPay QR Code สำหรับ Order ---
def generate_promptpay_qr_for_order(order_id):
    order = find_order_by_id(order_id)
    if not order:
        return {"success": False, "message": "ไม่พบคำสั่งซื้อ"}

    amount = float(order["amount"])
    if promptpay_qrcode is None:
        return {"success": False, "message": "ไม่พบไลบรารี promptpay กรุณาติดตั้งก่อน"}
    original_scannable_payload = promptpay_qrcode.generate_payload(os.getenv("PROMPTPAY_ID", "1103703685864"), amount)

    if not update_order_with_promptpay_payload_db(order_id, original_scannable_payload):
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

# --- ฟังก์ชันหลักในการตรวจสอบสลิปและอัปเดตสถานะ ---
def verify_payment_and_update_status(order_id, slip_image_path, payload_from_client):
    print(f"\n--- Processing payment for Order ID: {order_id} ---")
    print(f"Slip image path: {slip_image_path}")
    print(f"Payload (from client - original QR data): {payload_from_client}")

    try:
        # Find order
        order = find_order_by_id(order_id)
        if not order:
            print(f"❌ Error: Order ID {order_id} not found.")
            return {"success": False, "message": "ไม่พบคำสั่งซื้อ"}
        if order["order_status"] != 'pending':
            print(f"❌ Error: Order ID {order_id} is not pending. Current status: {order['order_status']}.")
            return {"success": False, "message": "คำสั่งซื้อนี้ดำเนินการไปแล้วหรือสถานะไม่ถูกต้อง"}
        # Check if an Ad already exists and its status
        ad = find_ad_by_order_id(order_id)
        if ad and ad['status'] != 'pending':
            print(f"❌ Error: Associated ad for Order ID {order_id} is not pending. Current ad status: {ad['status']}.")
            return {"success": False, "message": "โฆษณาสำหรับคำสั่งซื้อนี้มีการดำเนินการไปแล้ว"}
        # --- Call SlipOK API ---
        if not os.path.exists(slip_image_path):
            print(f"❌ Error: Slip image file not found at '{slip_image_path}'")
            return {"success": False, "message": "ไม่พบไฟล์รูปภาพสลิป"}
        with open(slip_image_path, 'rb') as img_file:
            files = {'files': img_file}
            form_data_for_slipok = {
                'log': 'true',
                'amount': str(float(order["amount"]))
            }
            headers = {
                "x-authorization": os.getenv("SLIP_OK_API_KEY", "SLIPOKKBE52WN"),
            }
            slipok_endpoint = os.getenv("SLIP_OK_API_ENDPOINT", "https://api.slipok.com/api/line/apikey/49130")
            print(f"Sending request to SlipOK API: {slipok_endpoint}")
            print(f"Headers sent: {headers}")
            print(f"Form Data sent to SlipOK: {form_data_for_slipok}")
            response = requests.post(slipok_endpoint, files=files, data=form_data_for_slipok, headers=headers, timeout=30)
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
    except requests.exceptions.Timeout:
        print(f"❌ Log: API Request Timeout: SlipOK API did not respond in time.")
        return {"success": False, "message": "ระบบตรวจสอบสลิปตอบกลับช้าเกินไป โปรดลองอีกครั้ง"}
    except requests.exceptions.HTTPError as e:
        print(f"❌ Log: Network or API HTTP Error (Unhandled by custom codes): {e}")
        try:
            error_details = response.json()
            print(f"    Error Details: {error_details}")
            return {"success": False, "message": f"เกิดข้อผิดพลาดในการเชื่อมต่อกับระบบตรวจสอบสลิป: {error_details.get('message', 'Unknown HTTP Error')}"}
        except Exception:
            return {"success": False, "message": f"เกิดข้อผิดพลาดในการเชื่อมต่อกับระบบตรวจสอบสลิป: {e}"}
    except requests.exceptions.RequestException as e:
        print(f"❌ Log: Network Error (e.g., DNS, connection refused): {e}")
        return {"success": False, "message": f"เกิดข้อผิดพลาดในการเชื่อมต่อกับระบบตรวจสอบสลิป: {e}"}
    except ValueError:
        print(f"❌ Log: Error: Could not parse amount from SlipOK response.")
        return {"success": False, "message": "รูปแบบยอดเงินจากระบบตรวจสอบสลิปไม่ถูกต้อง"}
    except Exception as e:
        print(f"❌ Log: An unexpected error occurred during SlipOK call: {e}")
        return {"success": False, "message": "เกิดข้อผิดพลาดภายใน"}
    # ตรวจสอบยอดเงิน
    if abs(slip_amount - float(order["amount"])) > 0.01:
        print(f"❌ Log: Amount mismatch. Order: {order['amount']}, Slip: {slip_amount}")
        return {"success": False, "message": f"ยอดเงินไม่ถูกต้อง (ต้องการ {order['amount']:.2f} บาท แต่ได้รับ {slip_amount:.2f} บาท)"}
    # --- เริ่มต้น Transaction เพื่ออัปเดตฐานข้อมูล ---
    try:
        # 1. อัปเดตสถานะ Order เป็น 'paid' พร้อมบันทึก Slip ID
        if not update_order_status_and_slip_info(order_id, "paid", slip_image_path, slip_transaction_id_from_api):
            raise Exception("Failed to update order status and slip info.")
        # 2. สร้างหรืออัปเดต Ad
        ad_id = None
        ad = find_ad_by_order_id(order_id)
        if ad: # ถ้ามี Ad อยู่แล้ว (อาจจะสถานะ pending)
            ad_id = ad['id']
            if not update_ad_status(ad_id, "paid"):
                raise Exception("Failed to update existing ad status to 'paid'.")
        else: # ถ้ายังไม่มี Ad ต้องสร้างใหม่
            ad_id = create_advertisement_db(order)
            if ad_id is None:
                raise Exception("Failed to create new advertisement.")
        print(f"✅ Transaction committed successfully for Order ID: {order_id} and Ad ID: {ad_id}")
        return {"success": True, "message": "ชำระเงินสำเร็จ กรุณารอแอดมินตรวจสอบ", "ad_id": ad_id}
    except Exception as e:
        db.session.rollback()
        print(f"❌ Log: Transaction failed for Order ID: {order_id}. Rolling back changes. Error: {e}")
        return {"success": False, "message": f"เกิดข้อผิดพลาดในการทำรายการ: {e}"}

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
    amount = db.Column(db.Numeric(10,2), nullable=False)
    order_status = db.Column(db.Enum('pending', 'paid', 'cancelled'), nullable=False, default='pending')
    slip_image = db.Column(db.String(255))
    created_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime)
    promptpay_qr_payload = db.Column(db.String(255))

class Ad(db.Model):
    __tablename__ = 'ads'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    order_id = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    link = db.Column(db.String(255))
    image = db.Column(db.String(255))
    status = db.Column(db.Enum('pending', 'paid', 'active', 'inactive'), nullable=False, default='pending')
    created_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime)
    expiration_date = db.Column(db.Date)
    admin_notes = db.Column(db.Text)

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
    """โหลดข้อมูลจากฐานข้อมูล MySQL และส่งคืนเป็น DataFrame"""
    try:
        engine = create_engine('mysql+mysqlconnector://bestpick_user:bestpick7890@localhost/reviewapp')
        
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
    """ทำให้คะแนนอยู่ในช่วง [0, 1]"""
    min_val, max_val = series.min(), series.max()
    if max_val > min_val:
        return (series - min_val) / (max_val - min_val)
    return series

def normalize_engagement(data, user_column='owner_id', engagement_column='PostEngagement'):
    """ปรับ Engagement ให้เหมาะสมตามผู้ใช้แต่ละคนให้อยู่ในช่วง [0, 1]"""
    data['NormalizedEngagement'] = data.groupby(user_column)[engagement_column].transform(lambda x: normalize_scores(x))
    return data

def analyze_comments(comments):
    """วิเคราะห์ความรู้สึกของคอมเมนต์ รองรับทั้งภาษาไทยและภาษาอังกฤษ"""
    sentiment_scores = []
    for comment in comments:
        try:
            if pd.isna(comment):
                sentiment_scores.append(0)
            else:
                # หากเป็นภาษาไทย ให้ tokenize ด้วย PyThaiNLP
                if any('\u0E00' <= char <= '\u0E7F' for char in comment):
                    tokenized_comment = ' '.join(word_tokenize(comment, engine='newmm'))
                else:
                    tokenized_comment = comment

                # คำนวณ Sentiment ด้วย TextBlob
                blob = TextBlob(tokenized_comment)
                polarity = blob.sentiment.polarity
                
                # กำหนด Sentiment Score
                if polarity > 0.5:
                    sentiment_scores.append(1)  # Sentiment บวก
                elif 0 < polarity <= 0.5:
                    sentiment_scores.append(0.5)  # Sentiment บวก
                elif -0.5 <= polarity < 0:
                    sentiment_scores.append(-0.5)  # Sentiment ลบ
                else:
                    sentiment_scores.append(-1)  # Sentiment ลบ
                    
        except Exception as e:
            sentiment_scores.append(0)  # หากเกิดข้อผิดพลาด ให้คะแนนเป็น 0
    return sentiment_scores

def create_content_based_model(data, text_column='Content', comment_column='Comments', engagement_column='PostEngagement'):
    """สร้างโมเดล Content-Based Filtering ด้วย TF-IDF และ KNN พร้อมแบ่งข้อมูล"""
    required_columns = [text_column, comment_column, engagement_column]
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"ข้อมูลขาดคอลัมน์ที่จำเป็น: {set(required_columns) - set(data.columns)}")

    # แบ่งข้อมูลเป็น train และ test
    train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

    # ใช้ TF-IDF เพื่อแปลงเนื้อหาของโพสต์เป็นเวกเตอร์
    tfidf = TfidfVectorizer(stop_words='english', max_features=6000, ngram_range=(1, 3), min_df=1, max_df=0.8)
    tfidf_matrix = tfidf.fit_transform(train_data[text_column].fillna(''))

    # ใช้ KNN เพื่อหาความคล้ายคลึงระหว่างโพสต์
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn.fit(tfidf_matrix)

    # วิเคราะห์ความรู้สึกจากความคิดเห็นใน train และ test sets
    train_data['SentimentScore'] = analyze_comments(train_data[comment_column])
    test_data['SentimentScore'] = analyze_comments(test_data[comment_column])

    # ปรับ Engagement ใน train set
    train_data = normalize_engagement(train_data)
    train_data['NormalizedEngagement'] = normalize_scores(train_data[engagement_column])
    train_data['WeightedEngagement'] = train_data['NormalizedEngagement'] + train_data['SentimentScore']

    # ปรับ Engagement ใน test set (กรณีใช้ในการประเมิน)
    test_data = normalize_engagement(test_data)

    joblib.dump(tfidf, 'TFIDF_Model.pkl')
    joblib.dump(knn, 'KNN_Model.pkl')
    return tfidf, knn, train_data, test_data

def create_collaborative_model(data, n_factors=150, n_epochs=70, lr_all=0.005, reg_all=0.5):
    """สร้างและฝึกโมเดล Collaborative Filtering พร้อมแบ่งข้อมูลเป็น training และ test set"""
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

def recommend_hybrid(user_id, train_data, test_data, collaborative_model, knn, tfidf, categories, alpha=0.50, beta=0.20):
    """
    แนะนำโพสต์โดยใช้ Hybrid Filtering รวม Collaborative, Content-Based และ Categories Adjustment
    :param alpha: น้ำหนักของคะแนน Collaborative (0 ถึง 1)
    :param beta: น้ำหนักของคะแนน Categories (0 ถึง 1)
    """
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha ต้องอยู่ในช่วง 0 ถึง 1")
    if not (0 <= beta <= 1):
        raise ValueError("Beta ต้องอยู่ในช่วง 0 ถึง 1")

    recommendations = []

    # ใช้ test_data ทั้งหมด
    for _, post in test_data.iterrows():
        # Collaborative Filtering
        collab_score = collaborative_model.predict(user_id, post['post_id']).est

        # Content-Based Filtering
        idx = train_data.index[train_data['post_id'] == post['post_id']].tolist()
        content_score = 0
        if idx:
            idx = idx[0]
            tfidf_vector = tfidf.transform([train_data.iloc[idx]['Content']])
            n_neighbors = min(20, knn._fit_X.shape[0])
            distances, indices = knn.kneighbors(tfidf_vector, n_neighbors=n_neighbors)
            content_score = np.mean([train_data.iloc[i]['NormalizedEngagement'] for i in indices[0]])

        # Categories Adjustment
        category_score = 0
        if categories:
            for category in categories:
                if category in post and post[category] == 1:  # เช็คว่าหมวดหมู่ตรงหรือไม่
                    category_score += 1

        # Normalize Category Score
        if categories:
            category_score /= len(categories)

        # Hybrid Score
        final_score = (alpha * collab_score) + ((1 - alpha) * content_score) + (beta * category_score)
        recommendations.append((post['post_id'], final_score))

    # Normalize Scores
    recommendations_df = pd.DataFrame(recommendations, columns=['post_id', 'score'])
    recommendations_df['normalized_score'] = normalize_scores(recommendations_df['score'])
    return recommendations_df.sort_values(by='normalized_score', ascending=False)['post_id'].tolist()

def split_and_rank_recommendations(recommendations, user_interactions):
    """แยกโพสต์ที่ผู้ใช้เคยโต้ตอบออกจากโพสต์ที่ยังไม่เคยดู และเรียงลำดับใหม่"""
    # แปลง recommendations ให้ไม่มีโพสต์ซ้ำ
    unique_recommendations = list(dict.fromkeys(recommendations))

    # แยกโพสต์ที่ผู้ใช้ยังไม่เคยดู และโพสต์ที่ผู้ใช้เคยโต้ตอบแล้ว
    unviewed_posts = [post_id for post_id in unique_recommendations if post_id not in user_interactions]
    viewed_posts = [post_id for post_id in unique_recommendations if post_id in user_interactions]

    # รวมโพสต์ที่ยังไม่เคยดู (unviewed) ก่อน ตามด้วยโพสต์ที่เคยดูแล้ว (viewed)
    final_recommendations = unviewed_posts + viewed_posts

    # พิมพ์ข้อมูลออกมา
    print("Unviewed Posts:", unviewed_posts)
    print("Viewed Posts:", viewed_posts)
    print("Final Recommendations (ordered):", final_recommendations)

    return final_recommendations

# Cache สำหรับเก็บคำแนะนำของผู้ใช้
recommendation_cache = {}
cache_expiry_time = 10  # หน่วยเป็นวินาที (10 วินาที)

# ฟังก์ชันสำหรับ clear cache
def clear_cache():
    """เคลียร์ cache ทุกๆ 10 วินาที"""
    global recommendation_cache
    while True:
        time.sleep(cache_expiry_time)  # รอ 10 วินาที
        recommendation_cache = {}
        print("Cache cleared automatically.")

# สร้าง thread สำหรับ clear cache
threading.Thread(target=clear_cache, daemon=True).start()

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
                "valid_photos": photo_urls,
                "suggestion": "กรุณาลบภาพที่ไม่เหมาะสมออกจากโพสต์และลองใหม่อีกครั้ง"
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
        existing_photos = request.form.getlist('existing_photos')
        existing_videos = request.form.getlist('existing_videos')
        photos = request.files.getlist('photo')
        videos = request.files.getlist('video')

        if not user_id:
            return jsonify({"error": "You are not authorized to update this post"}), 403

        photo_urls = existing_photos if existing_photos else []
        video_urls = existing_videos if existing_videos else []
        invalid_photos = []

        for photo in photos:
            if not photo or not photo.filename:
                continue
                
            try:
                filename = secure_filename(photo.filename)
                photo_path = os.path.join(UPLOAD_FOLDER, filename)
                photo.save(photo_path)
                
                is_nude, result = nude_predict_image(photo_path)
                
                if is_nude:
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
                    photo_urls.append(f'/uploads/{filename}')
                    
            except Exception as e:
                print(f"Error processing photo {photo.filename}: {e}")
                # หากเกิดข้อผิดพลาด ให้ลบไฟล์และแจ้งเตือน
                if os.path.exists(photo_path):
                    os.remove(photo_path)
                invalid_photos.append({
                    "filename": photo.filename,
                    "reason": "ไม่สามารถประมวลผลภาพได้",
                    "details": {"error": str(e)}
                })
        
        # หากมีภาพที่ไม่เหมาะสม ให้แจ้งเตือนและไม่อัปเดต post
        if invalid_photos:
            return jsonify({
                "status": "warning",
                "message": "กรุณาเปลี่ยนภาพแล้วลองใหม่อีกครั้ง",
                "invalid_photos": invalid_photos,
                "valid_photos": photo_urls,
                "suggestion": "กรุณาลบภาพที่ไม่เหมาะสมออกจากโพสต์และลองใหม่อีกครั้ง"
            }), 400

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
                    category = :category, photo_url = :photo_urls, video_url = :video_urls, 
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
                'category': CategoryID,
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

        # หาก cache มีผลลัพธ์สำหรับ user_id นี้ ให้ใช้ผลลัพธ์จาก cache
        if user_id in recommendation_cache:
            print(f"Returning cached recommendations for user_id: {user_id}")
            return jsonify(recommendation_cache[user_id])

        # โหลดข้อมูลจากฐานข้อมูล
        content_based_data, collaborative_data = load_data_from_db()

        # สร้างคอลัมน์ 'NormalizedEngagement' หากยังไม่มี
        if 'NormalizedEngagement' not in content_based_data.columns:
            content_based_data = normalize_engagement(content_based_data, user_column='owner_id', engagement_column='PostEngagement')

        # Load pre-trained models
        try:
            knn = joblib.load('KNN_Model.pkl')
            collaborative_model = joblib.load('Collaborative_Model.pkl')
            tfidf = joblib.load('TFIDF_Model.pkl')
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
            return jsonify({"error": "Model files not found"}), 500

        # สร้างหมวดหมู่
        categories = [
            'Gadget', 'Smartphone', 'Laptop', 'Smartwatch', 'Headphone', 'Tablet', 'Camera', 'Drone',
            'Home_Appliance', 'Gaming_Console', 'Wearable_Device', 'Fitness_Tracker', 'VR_Headset',
            'Smart_Home', 'Power_Bank', 'Bluetooth_Speaker', 'Action_Camera', 'E_Reader',
            'Desktop_Computer', 'Projector'
        ]

        # คำนวณคำแนะนำใหม่
        recommendations = recommend_hybrid(
            user_id, content_based_data, collaborative_data,
            collaborative_model, knn, tfidf, categories,
            alpha=0.8, beta=0.2  # เพิ่ม beta สำหรับ categories
        )

        if not recommendations:
            return jsonify({"error": "No recommendations found"}), 404

        # แยกโพสต์ที่ผู้ใช้เคยโต้ตอบ และยังไม่เคยดู
        user_interactions = collaborative_data[collaborative_data['user_id'] == user_id]['post_id'].tolist()
        final_recommendations = split_and_rank_recommendations(recommendations, user_interactions)

        # Query for post details
        placeholders = ', '.join([f':id_{i}' for i in range(len(final_recommendations))])
        query = text(f"""
            SELECT posts.*, users.username, users.picture,
                   (SELECT COUNT(*) FROM likes WHERE post_id = posts.id AND user_id = :user_id) AS is_liked
            FROM posts 
            JOIN users ON posts.user_id = users.id
            WHERE posts.status = 'active' AND posts.id IN ({placeholders})
        """)

        params = {'user_id': user_id, **{f'id_{i}': post_id for i, post_id in enumerate(final_recommendations)}}
        result = db.session.execute(query, params).fetchall()
        posts = [row._mapping for row in result]

        # ใช้ final_recommendations เพื่อรักษาลำดับที่แนะนำ
        sorted_posts = sorted(posts, key=lambda x: final_recommendations.index(x['id']))

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

        # บันทึกผลลัพธ์ลงใน cache
        recommendation_cache[user_id] = output

        return jsonify(output)

    except KeyError as e:
        print(f"KeyError in recommend function: {e}")
        return jsonify({"error": f"KeyError: {e}"}), 500
    except Exception as e:
        print("Error in recommend function:", e)
        return jsonify({"error": "Internal Server Error"}), 500

# --- API Routes สำหรับ Slip ---
@app.route('/api/generate-qrcode/<int:order_id>', methods=['GET'])
def api_generate_qrcode(order_id):
    result = generate_promptpay_qr_for_order(order_id)
    if not result['success']:
        return jsonify(result), 400
    return jsonify({
        'success': True,
        'order_id': order_id,
        'qrcode_base64': result.get('qrcode_base64'),
        'promptpay_payload': result.get('payload')
    })

@app.route('/api/verify-slip/<int:order_id>', methods=['POST'])
def api_verify_slip(order_id):
    if 'slip_image' not in request.files:
        return jsonify({'success': False, 'message': 'ต้องอัปโหลดไฟล์ slip_image'}), 400
    file = request.files['slip_image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'ไม่ได้เลือกไฟล์'}), 400
    if 'payload' not in request.form:
        return jsonify({'success': False, 'message': 'ต้องระบุ payload (QR Code)'}), 400
    payload = request.form['payload']
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    slip_dir = 'Slip'
    if not os.path.exists(slip_dir):
        os.makedirs(slip_dir)
    save_path = os.path.join(slip_dir, unique_filename)
    file.save(save_path)
    print(f"✅ Slip image uploaded to {save_path}")
    print(f"✅ Payload from client (QR Code data): {payload}")
    result = verify_payment_and_update_status(order_id, save_path, payload)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)