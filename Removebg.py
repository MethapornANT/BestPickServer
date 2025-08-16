from rembg import remove
from PIL import Image

# เปิดรูปต้นฉบับ
input_path = "ChatGPT Image 14 ส.ค. 2568 20_15_40.png"
output_path = "output.png"

# โหลดภาพและลบพื้นหลัง
input_image = Image.open(input_path)
output_image = remove(input_image)

# บันทึกไฟล์ (จะได้ PNG โปร่งใส)
output_image.save(output_path)
