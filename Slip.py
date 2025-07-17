import mysql.connector
from datetime import datetime, timedelta
import requests
import os
import qrcode
from qrcode.constants import ERROR_CORRECT_H
import uuid
from flask import Flask, jsonify, request
import base64
import io
from promptpay import qrcode as promptpay_qrcode

app = Flask(__name__)

# --- ตั้งค่าการเชื่อมต่อฐานข้อมูลของคุณ ---
# ใช้ localhost สำหรับการพัฒนาในเครื่อง
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',    
    'password': '1234', 
    'database': 'bestpick'    
}

# --- ตั้งค่า SlipOK API (สำคัญ: ใน Production ควรเก็บใน Environment Variables) ---
SLIP_OK_API_ENDPOINT = os.getenv("SLIP_OK_API_ENDPOINT", "https://api.slipok.com/api/line/apikey/49130")
SLIP_OK_API_KEY = os.getenv("SLIP_OK_API_KEY", "SLIPOKKBE52WN") 

# --- PromptPay ID ของผู้รับ ---
# ควรอ่านจาก Environment Variable ใน Production
PROMPTPAY_RECEIVER_ID = os.getenv("PROMPTPAY_ID", "1103703685864")

# --- ฟังก์ชันช่วยเหลือในการเชื่อมต่อฐานข้อมูล ---
def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return None

# --- ฟังก์ชัน DB Interactions ---

def find_order_by_id(order_id, conn=None):
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        if conn is None: return None
        close_conn = True
    
    cursor = conn.cursor(dictionary=True)
    try:
        query = "SELECT id, user_id, amount, order_status, promptpay_qr_payload FROM orders WHERE id = %s"
        cursor.execute(query, (order_id,))
        order = cursor.fetchone()
        return order
    except mysql.connector.Error as err:
        print(f"Error finding order by ID: {err}")
        return None
    finally:
        cursor.close()
        if close_conn and conn.is_connected():
            conn.close()

def find_ad_by_order_id(order_id, conn=None):
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        if conn is None: return None
        close_conn = True
    
    cursor = conn.cursor(dictionary=True)
    try:
        query = "SELECT id, status, expiration_date FROM ads WHERE order_id = %s"
        cursor.execute(query, (order_id,))
        ad = cursor.fetchone()
        return ad
    except mysql.connector.Error as err:
        print(f"Error finding ad by order ID: {err}")
        return None
    finally:
        cursor.close()
        if close_conn and conn.is_connected():
            conn.close()

def update_order_status_and_slip_info(order_id, new_status, slip_image_path, slip_transaction_id, conn):
    cursor = conn.cursor()
    try:
        query = """
            UPDATE orders
            SET order_status = %s, slip_image = %s, slip_transaction_id = %s, updated_at = NOW()
            WHERE id = %s
        """
        cursor.execute(query, (new_status, slip_image_path, slip_transaction_id, order_id))
        print(f"✅ Order ID: {order_id} status updated to '{new_status}' with slip info.")
        return True
    except mysql.connector.Error as err:
        print(f"Error updating order status for ID {order_id}: {err}")
        return False
    finally:
        cursor.close()

def update_ad_status(ad_id, new_status, conn):
    cursor = conn.cursor()
    try:
        query = "UPDATE ads SET status = %s, updated_at = NOW() WHERE id = %s"
        cursor.execute(query, (new_status, ad_id))
        print(f"✅ Ad ID: {ad_id} status updated to '{new_status}'.")
        return True
    except mysql.connector.Error as err:
        print(f"Error updating ad status for ID {ad_id}: {err}")
        return False
    finally:
        cursor.close()

def update_order_with_promptpay_payload_db(order_id, payload_to_store_in_db, conn=None):
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        if conn is None: return False
        close_conn = True

    cursor = conn.cursor()
    try:
        query = """
            UPDATE orders
            SET promptpay_qr_payload = %s, updated_at = NOW()
            WHERE id = %s
        """
        cursor.execute(query, (payload_to_store_in_db, order_id))
        if close_conn:
            conn.commit()
        print(f"✅ Order ID: {order_id} updated with PromptPay payload.")
        return True
    except mysql.connector.Error as err:
        print(f"Error updating order with PromptPay payload: {err}")
        if close_conn:
            conn.rollback()
        return False
    finally:
        cursor.close()
        if close_conn and conn.is_connected():
            conn.close()

def create_advertisement_db(order_data, conn):
    cursor = conn.cursor()
    try:
        now = datetime.now()
        # สถานะเริ่มต้นของโฆษณาเมื่อสร้างคือ 'paid' (รอ Admin Approve)
        query = """
            INSERT INTO ads
            (user_id, order_id, title, content, status, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        default_title = f"Advertisement for Order {order_data['id']}"
        default_content = "This is a new advertisement pending admin approval after payment."

        cursor.execute(query, (
            order_data['user_id'],
            order_data['id'],
            default_title,
            default_content,
            'paid', # สถานะเริ่มต้นเป็น 'paid' รอ Admin Approve
            now
        ))
        ad_id = cursor.lastrowid
        print(f"🚀 Advertisement ID: {ad_id} created for Order ID: {order_data['id']} with status 'paid'.")
        return ad_id
    except mysql.connector.Error as err:
        print(f"Error creating advertisement: {err}")
        return None
    finally:
        cursor.close()

# --- ฟังก์ชันสร้าง PromptPay QR Code สำหรับ Order ---
def generate_promptpay_qr_for_order(order_id):
    order = find_order_by_id(order_id)
    if not order:
        return {"success": False, "message": "ไม่พบคำสั่งซื้อ"}

    amount = float(order["amount"])
    
    original_scannable_payload = promptpay_qrcode.generate_payload(PROMPTPAY_RECEIVER_ID, amount)

    # Note: Using a separate connection for this simple update outside main transaction flow
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
    if hasattr(img, 'get_image'): # Pillow Image
        img.get_image().save(buffered, format="PNG")
    else: # qrcode library's image
        img.save(buffered, format="PNG")
    
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return {"success": True, "message": "สร้าง QR Code สำเร็จ", "qrcode_base64": img_b64, "payload": original_scannable_payload}

# --- ฟังก์ชันหลักในการตรวจสอบสลิปและอัปเดตสถานะ ---
def verify_payment_and_update_status(order_id, slip_image_path, payload_from_client):
    print(f"\n--- Processing payment for Order ID: {order_id} ---")
    print(f"Slip image path: {slip_image_path}")
    print(f"Payload (from client - original QR data): {payload_from_client}")

    conn = None # Initialize conn to None for error handling
    try:
        # Find order (use local connection as this is outside the main transaction yet)
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
    conn = get_db_connection()
    if conn is None:
        return {"success": False, "message": "ไม่สามารถเชื่อมต่อฐานข้อมูลได้เพื่อทำรายการ"}
    try:
        conn.start_transaction()
        
        # 1. อัปเดตสถานะ Order เป็น 'paid' พร้อมบันทึก Slip ID
        if not update_order_status_and_slip_info(order_id, "paid", slip_image_path, slip_transaction_id_from_api, conn):
            raise Exception("Failed to update order status and slip info.")
        
        # 2. สร้างหรืออัปเดต Ad
        ad_id = None
        if ad: # ถ้ามี Ad อยู่แล้ว (อาจจะสถานะ pending)
            ad_id = ad['id']
            if not update_ad_status(ad_id, "paid", conn): # อัปเดตสถานะ Ad เป็น 'paid'
                raise Exception("Failed to update existing ad status to 'paid'.")
        else: # ถ้ายังไม่มี Ad ต้องสร้างใหม่
            ad_id = create_advertisement_db(order, conn=conn) # ส่ง conn เข้าไปเพื่อให้อยู่ใน Transaction เดียวกัน
            if ad_id is None:
                raise Exception("Failed to create new advertisement.")
        
        conn.commit()
        print(f"✅ Transaction committed successfully for Order ID: {order_id} and Ad ID: {ad_id}")
        return {"success": True, "message": "ชำระเงินสำเร็จ กรุณารอแอดมินตรวจสอบ", "ad_id": ad_id}

    except Exception as e:
        print(f"❌ Log: Transaction failed for Order ID: {order_id}. Rolling back changes. Error: {e}")
        conn.rollback()
        return {"success": False, "message": f"เกิดข้อผิดพลาดในการทำรายการ: {e}"}
    finally:
        if conn and conn.is_connected():
            conn.close()

# --- API Routes ---
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
    Slip_dir = 'Slip'
    if not os.path.exists(Slip_dir):
        os.makedirs(Slip_dir)
    save_path = os.path.join(Slip_dir, unique_filename)
    file.save(save_path)
    print(f"✅ Slip image uploaded to {save_path}")
    print(f"✅ Payload from client (QR Code data): {payload}")
    
    result = verify_payment_and_update_status(order_id, save_path, payload)
    return jsonify(result)

# --- Main execution ---
if __name__ == '__main__':
    conn_test = get_db_connection()
    if conn_test:
        print("Database connection successful!")
        conn_test.close()
    else:
        print("Failed to connect to the database. Please check DB_CONFIG.")
        exit()

    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    if not os.path.exists('Slip'):
        os.makedirs('Slip')

    # รัน Flask บนพอร์ต 5000 (สำหรับ Slip Processing API)
    app.run(port=5000, debug=True)