import qrcode
from promptpay import qrcode as promptpay_qrcode # Import the qrcode module from promptpay library

# --- PromptPay Details ---
# เลข PromptPay ID (เลขบัตรประชาชน 13 หลัก)
# ต้องใส่เป็น string เพราะเป็น ID ไม่ใช่ตัวเลขที่เอาไปคำนวณ
promptpay_id = "1103703685864"
amount = 1.00  # จำนวนเงินที่ต้องการโอน (1 บาท)

# --- Generate PromptPay QR Code String ---
# promptpay_qrcode.generate_payload() จะสร้างข้อมูล payload ที่ถูกต้องตามมาตรฐาน
# โดยจะรับ PromptPay ID และ amount เข้าไป
# ไลบรารีจะจัดการเรื่องประเภท ID (เบอร์โทร/บัตรประชาชน) และ CRC ให้โดยอัตโนมัติ
# ตรวจสอบ: หาก ID ขึ้นต้นด้วย 06, 08, 09 (เบอร์มือถือ) จะถือเป็น Mobile Number ID
# หากเป็นตัวเลข 13 หลักอื่นๆ จะถือเป็น National ID หรือ e-Wallet ID
payload = promptpay_qrcode.generate_payload(promptpay_id, amount)

print(f"Generated PromptPay Payload using promptpay library: {payload}")

# --- Create QR Code Image ---
# ใช้ qrcode library ปกติ ในการสร้างภาพจาก payload ที่ได้มา
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H, # แนะนำให้ใช้ระดับสูง
    box_size=10,
    border=4,
)
qr.add_data(payload)
qr.make(fit=True)

# --- Create and Save Image ---
img = qr.make_image(fill_color="black", back_color="white")
file_name = "promptpay_qr_1103703685864_1baht_PROMPT_PAY_LIB.png"
img.save(file_name)

print(f"\nQR Code generated and saved as {file_name}")
print(f"**กรุณาลองสแกน QR Code นี้ด้วยแอปธนาคารของคุณอีกครั้งครับ**")
print(f"**ควรจะแสดงข้อมูล PromptPay ID (เลขบัตรประชาชน): {promptpay_id} และยอดเงิน 1 บาทอย่างถูกต้อง**")
print("ครั้งนี้ใช้ไลบรารีที่ออกแบบมาสำหรับ PromptPay โดยตรง หวังว่าจะใช้งานได้จริงแน่นอนครับ!")