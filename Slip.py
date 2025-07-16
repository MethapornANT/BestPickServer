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
from promptpay import qrcode as promptpay_qrcode # ย้ายมาไว้ข้างบนเพื่อให้เห็นชัดเจน

app = Flask(__name__)

# --- ตั้งค่าการเชื่อมต่อฐานข้อมูลของคุณ ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',     # ใส่ชื่อผู้ใช้งาน MySQL ของคุณ
    'password': '1234', # ใส่รหัสผ่าน MySQL ของคุณ
    'database': 'reviewapptest'      # ใส่ชื่อฐานข้อมูลของคุณ
}

# --- ตั้งค่า SlipOK API (สำคัญ: ใน Production ควรเก็บใน Environment Variables) ---
SLIP_OK_API_ENDPOINT = "https://api.slipok.com/api/line/apikey/49130"
SLIP_OK_API_KEY = "SLIPOKKBE52WN" 

# --- ฟังก์ชันช่วยเหลือในการเชื่อมต่อฐานข้อมูล ---
def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return None

# --- ฟังก์ชัน DB Interactions ---

def find_order_by_id(order_id):
    conn = get_db_connection()
    if conn is None:
        return None
    cursor = conn.cursor(dictionary=True)
    try:
        # แก้ไข: ลบ slip_transaction_id ออกจาก query
        query = "SELECT id, user_id, amount, order_status, promptpay_qr_payload FROM orders WHERE id = %s"
        cursor.execute(query, (order_id,))
        order = cursor.fetchone()
        return order
    except mysql.connector.Error as err:
        print(f"Error finding order by ID: {err}")
        return None
    finally:
        cursor.close()
        conn.close()

def update_order_with_promptpay_payload(order_id, payload_to_store_in_db):
    conn = get_db_connection()
    if conn is None:
        return False
    cursor = conn.cursor()
    try:
        query = """
            UPDATE orders
            SET promptpay_qr_payload = %s, updated_at = NOW()
            WHERE id = %s
        """
        cursor.execute(query, (payload_to_store_in_db, order_id))
        conn.commit()
        print(f"✅ Order ID: {order_id} updated with PromptPay payload '{payload_to_store_in_db}' in promptpay_qr_payload.")
        return True
    except mysql.connector.Error as err:
        print(f"Error updating order with PromptPay payload: {err}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

def create_advertisement(order_data, package_duration_days):
    conn = get_db_connection()
    if conn is None:
        return None
    cursor = conn.cursor()
    try:
        now = datetime.now()
        expires_at = (now + timedelta(days=package_duration_days)).date()

        query = """
            INSERT INTO ads
            (user_id, order_id, title, content, status, created_at, expiration_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        default_title = f"Advertisement for Order {order_data['id']}"
        default_content = "This is a new advertisement activated by payment."

        cursor.execute(query, (
            order_data['user_id'],
            order_data['id'],
            default_title,
            default_content,
            'active',
            now,
            expires_at
        ))
        conn.commit()
        ad_id = cursor.lastrowid
        print(f"🚀 Advertisement ID: {ad_id} created for Order ID: {order_data['id']}. Expires at: {expires_at.isoformat()}")
        return ad_id
    except mysql.connector.Error as err:
        print(f"Error creating advertisement: {err}")
        conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()

# --- ฟังก์ชันสร้าง PromptPay QR Code สำหรับ Order ---
def generate_promptpay_qr_for_order(order_id):
    order = find_order_by_id(order_id)
    if not order:
        return {"success": False, "message": "ไม่พบคำสั่งซื้อ"}

    amount = float(order["amount"])
    promptpay_id = "1103703685864" # PromptPay ID ของผู้รับ ต้องตรงกับ SlipOK dashboard
    
    # Generate the ORIGINAL PromptPay payload (for QR Code image)
    original_scannable_payload = promptpay_qrcode.generate_payload(promptpay_id, amount)

    # แก้ไข: ไม่ต้องใส่ order_id นำหน้า payload แล้ว และไม่ต้องกังวลเรื่อง uniqueness ที่ฝั่งนี้
    payload_to_store_in_db = original_scannable_payload

    if not update_order_with_promptpay_payload(order_id, payload_to_store_in_db):
        return {"success": False, "message": "ไม่สามารถบันทึกข้อมูล QR Code ลงฐานข้อมูลได้"}
    
    # Note: We print the DB-stored payload here for logging
    # แก้ไข: ปรับข้อความ Log
    print(f"✅ Generated PromptPay payload (stored in DB): {payload_to_store_in_db}")
    
    # Use the ORIGINAL payload for QR code generation
    qr = qrcode.QRCode(
        version=1,
        error_correction=ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(original_scannable_payload) # Use original payload here
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    qr_path = f"qrcode_order_{order_id}.png"
    if hasattr(img, 'get_image'):
        img = img.get_image()
    img.save(qr_path)
    
    # Return the ORIGINAL payload to the client, as they will send this back for verification
    return {"success": True, "message": "สร้าง QR Code สำเร็จ", "qr_path": qr_path, "payload": original_scannable_payload}

# --- ฟังก์ชันหลักในการตรวจสอบสลิป (ปรับปรุงตาม SlipOK API Guide และความต้องการผู้ใช้) ---

def verify_payment_and_activate_ad(order_id, slip_image_path, payload_from_client): # Renamed payload to payload_from_client for clarity
    print(f"\n--- Processing payment for Order ID: {order_id} ---")
    print(f"Slip image path: {slip_image_path}")
    print(f"Payload (from client - original QR data): {payload_from_client}")

    order = find_order_by_id(order_id)
    if not order:
        print(f"❌ Error: Order ID {order_id} not found.")
        return {"success": False, "message": "ไม่พบคำสั่งซื้อ"}
    if order["order_status"] != 'pending':
        print(f"❌ Error: Order ID {order_id} is not pending. Current status: {order['order_status']}.")
        return {"success": False, "message": "คำสั่งซื้อนี้ดำเนินการไปแล้วหรือสถานะไม่ถูกต้อง"}
    
    try:
        if not os.path.exists(slip_image_path):
            print(f"❌ Error: Slip image file not found at '{slip_image_path}'")
            return {"success": False, "message": "ไม่พบไฟล์รูปภาพสลิป"}

        with open(slip_image_path, 'rb') as img_file:
            files = {'files': img_file}
            form_data_for_slipok = {
                'log': 'true',
                'amount': str(float(order["amount"])) # แปลง amount เป็น string สำหรับ form-data
            }
            headers = {
                "x-authorization": SLIP_OK_API_KEY,
            }
            print(f"Sending request to SlipOK API: {SLIP_OK_API_ENDPOINT}")
            print(f"Headers sent: {headers}")
            print(f"Form Data sent to SlipOK: {form_data_for_slipok}")

            response = requests.post(SLIP_OK_API_ENDPOINT, files=files, data=form_data_for_slipok, headers=headers, timeout=30)
            response.raise_for_status() 

            # เพิ่มบรรทัดนี้เพื่อดู Response ทั้งหมด
            print(f"DEBUG: Full SlipOK response text: {response.text}")

            slip_ok_response_data = response.json()
            print(f"Received response from SlipOK: {slip_ok_response_data}")

            # แก้ไข: จัดการ Error Code และข้อความให้เฉพาะเจาะจงมากขึ้น
            if not slip_ok_response_data.get("success"):
                error_code = str(slip_ok_response_data.get("code")) # ตรวจสอบให้แน่ใจว่าเป็น string สำหรับการเปรียบเทียบ
                error_message = slip_ok_response_data.get("message", "Unknown error from SlipOK API")
                
                print(f"❌ Log: Error from SlipOK API (Code: {error_code}): {error_message}")

                if error_code == "1002": # "Authorization Header ไม่ถูกต้อง" 
                    return {"success": False, "message": f"การตรวจสอบสลิปไม่สำเร็จ: API Key ไม่ถูกต้องหรือไม่ได้รับอนุญาต ({error_message})"}
                elif error_code == "1012": # "สลิปซ้ำ (Duplicate Slip)"
                    print(f"⚠️ Log: User attempted to use a duplicate slip for Order ID {order_id}. SlipOK reports: {error_message}")
                    return {"success": False, "message": f"สลิปนี้ถูกใช้งานไปแล้ว กรุณาเปลี่ยนสลิป ({error_message})"}
                elif error_code == "1013": # "ยอดที่ส่งมาไม่ตรงกับยอดสลิป"
                    print(f"⚠️ Log: Amount mismatch reported by SlipOK for Order ID {order_id}. SlipOK reports: {error_message}")
                    return {"success": False, "message": f"ยอดเงินในสลิปไม่ตรงกับยอดที่ต้องการ กรุณาตรวจสอบ ({error_message})"}
                elif error_code == "1014": # "บัญชีผู้รับไม่ตรงกับบัญชีหลักของร้าน" 
                    return {"success": False, "message": f"การตรวจสอบสลิปไม่สำเร็จ: บัญชีผู้รับในสลิปไม่ตรงกับบัญชีที่ตั้งค่าใน SlipOK ({error_message})"}
                elif error_code == "1006": # "รูปภาพไม่ถูกต้อง"
                    return {"success": False, "message": f"รูปภาพสลิปไม่ถูกต้อง หรือไม่สามารถอ่านได้ ({error_message})"}
                elif error_code == "1007": # "ไม่มี QR Code ในรูปภาพ"
                    return {"success": False, "message": f"ไม่พบ QR Code ในรูปภาพสลิป หรือ QR Code หมดอายุ ({error_message})"}
                elif error_code == "1008": # "QR Code ไม่ใช่สำหรับชำระเงิน"
                    return {"success": False, "message": f"QR Code ในสลิปไม่ใช่สำหรับการชำระเงิน ({error_message})"}
                elif error_code == "1010": # "สลิปมี Delay"
                    return {"success": False, "message": f"สลิปนี้ยังไม่ถูกบันทึกในระบบธนาคาร กรุณารอซักครู่แล้วลองใหม่ ({error_message})"}
                
                # ข้อผิดพลาดอื่นๆ ที่ไม่ตรงกับรหัสข้างต้น
                return {"success": False, "message": f"การตรวจสอบสลิปไม่สำเร็จ: {error_message}"}

            # แก้ไข: ตรวจสอบโครงสร้างข้อมูลที่ยืดหยุ่นมากขึ้น และใช้ transRef แทน transactionId
            slipok_data = slip_ok_response_data.get("data")
            if not slipok_data:
                print(f"❌ Log: Unexpected response format from SlipOK API: 'data' field is missing or empty.")
                return {"success": False, "message": "รูปแบบข้อมูลจากระบบตรวจสอบสลิปไม่ถูกต้อง (ไม่พบข้อมูลสลิป)"}

            # แก้ไข: ใช้ .get() เพื่อเข้าถึงคีย์อย่างปลอดภัย
            slip_transaction_id_from_api = slipok_data.get("transRef") # เปลี่ยนจาก "transactionId" เป็น "transRef"
            slip_amount = float(slipok_data.get("amount", 0.0)) # ใส่ค่าเริ่มต้น 0.0 เผื่อไม่มี amount

            if not slip_transaction_id_from_api:
                print(f"❌ Log: Missing 'transRef' in SlipOK 'data' object.")
                return {"success": False, "message": "รูปแบบข้อมูลจากระบบตรวจสอบสลิปไม่ถูกต้อง (ไม่พบ Transaction ID)"}

    except requests.exceptions.Timeout:
        print(f"❌ Log: API Request Timeout: SlipOK API did not respond in time.")
        return {"success": False, "message": "ระบบตรวจสอบสลิปตอบกลับช้าเกินไป โปรดลองอีกครั้ง"}
    except requests.exceptions.HTTPError as e:
        # ข้อผิดพลาด 4xx/5xx ที่ไม่ได้ถูกดักด้วย response.json().get("success") (เช่น ถ้า API ส่งกลับมาเป็น HTML แทน JSON)
        print(f"❌ Log: Network or API HTTP Error (Unhandled by custom codes): {e}")
        try:
            # พยายามอ่าน response body อีกครั้งเผื่อมีรายละเอียดที่ไม่ได้ถูกดัก
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

    # แก้ไข: ลบการตรวจสอบสลิปซ้ำภายในระบบด้วย find_order_by_slip_id() ออกไป
    # if find_order_by_slip_id(slip_transaction_id_from_api):
    #     print(f"❌ Security Alert: Slip Transaction ID '{slip_transaction_id_from_api}' has already been used (internal check)!")
    #     return {"success": False, "message": "สลิปนี้ถูกใช้งานไปแล้ว (ตรวจพบโดยระบบภายใน)"}

    if abs(slip_amount - float(order["amount"])) > 0.01:
        print(f"❌ Log: Amount mismatch. Order: {order['amount']}, Slip: {slip_amount}")
        return {"success": False, "message": f"ยอดเงินไม่ถูกต้อง (ต้องการ {order['amount']:.2f} บาท แต่ได้รับ {slip_amount:.2f} บาท)"}

    conn = get_db_connection()
    if conn is None:
        return {"success": False, "message": "ไม่สามารถเชื่อมต่อฐานข้อมูลได้เพื่อทำรายการ"}
    try:
        conn.start_transaction()
        cursor = conn.cursor()
        # แก้ไข: ลบ slip_transaction_id ออกจาก query UPDATE
        query = """
            UPDATE orders
            SET order_status = %s, slip_image = %s, updated_at = NOW()
            WHERE id = %s
        """
        # แก้ไข: ลบ slip_transaction_id_from_api ออกจาก parameters
        cursor.execute(query, ("paid", slip_image_path, order_id))
        conn.commit()
        print(f"✅ Order ID: {order_id} updated to 'paid' with slip image.")

        package_duration_days = 30
        ad_id = create_advertisement(order, package_duration_days)
        if ad_id is None:
            raise Exception("Failed to create advertisement.")
        conn.commit()
        print(f"✅ Transaction committed successfully for Order ID: {order_id}")
        return {"success": True, "message": "ชำระเงินสำเร็จกรุณารอการตรวจสอบ", "ad_id": ad_id}
    except Exception as e:
        print(f"❌ Log: Transaction failed for Order ID: {order_id}. Rolling back changes. Error: {e}")
        conn.rollback()
        return {"success": False, "message": f"เกิดข้อผิดพลาดในการทำรายการ: {e}"}
    finally:
        if conn and conn.is_connected():
            conn.close()

# --- ทดสอบการใช้งาน ---
if __name__ == "__main__":
    conn_test = get_db_connection()
    if conn_test:
        print("Database connection successful!")
        conn_test.close()
    else:
        print("Failed to connect to the database. Please check DB_CONFIG.")
        exit()

@app.route('/api/generate-qrcode/<int:order_id>', methods=['GET'])
def api_generate_qrcode(order_id):
    result = generate_promptpay_qr_for_order(order_id)
    if not result['success']:
        return jsonify(result), 400
    with open(result['qr_path'], 'rb') as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return jsonify({
        'success': True,
        'order_id': order_id,
        'qrcode_base64': img_b64,
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
    
    Slip_dir = 'Slip'
    if not os.path.exists(Slip_dir):
        os.makedirs(Slip_dir)
    save_path = os.path.join(Slip_dir, file.filename)
    file.save(save_path)
    print(f"✅ Slip image uploaded to {save_path}")
    print(f"✅ Payload from client (QR Code data): {payload}")
    
    result = verify_payment_and_activate_ad(order_id, save_path, payload)
    return jsonify(result)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(port=5000, debug=True)