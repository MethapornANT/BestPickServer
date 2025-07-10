from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import json

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# โหลดโมเดลและ processor
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
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    hentai_score = probs[1] * 100
    porn_score = probs[3] * 100
    return hentai_score > 20 or porn_score > 20, {id2label[str(i)]: round(probs[i]*100, 2) for i in range(len(probs))}

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
        for photo in photos:
            if not photo or not photo.filename:
                continue
            filename = secure_filename(photo.filename)
            photo_path = os.path.join(UPLOAD_FOLDER, filename)
            photo.save(photo_path)
            is_nude, result = nude_predict_image(photo_path)
            if is_nude:
                os.remove(photo_path)
                return jsonify({"error": "พบภาพโป๊ (Hentai หรือ Pornography > 20%) กรุณาลบภาพดังกล่าวออกจากโพสต์", "result": result}), 400
            photo_urls.append(f'/uploads/{filename}')

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

        # mockup: post_id = 1
        return jsonify({
            "post_id": 1,
            "user_id": user_id,
            "content": content,
            "category": category,
            "Title": title,
            "ProductName": product_name,
            "video_urls": video_urls,
            "photo_urls": photo_urls
        }), 201

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

        for photo in photos:
            if not photo or not photo.filename:
                continue
            filename = secure_filename(photo.filename)
            photo_path = os.path.join(UPLOAD_FOLDER, filename)
            photo.save(photo_path)
            is_nude, result = nude_predict_image(photo_path)
            if is_nude:
                os.remove(photo_path)
                return jsonify({"error": "พบภาพโป๊ (Hentai หรือ Pornography > 20%) กรุณาลบภาพดังกล่าวออกจากโพสต์", "result": result}), 400
            photo_urls.append(f'/uploads/{filename}')

        for video in videos:
            if not video or not video.filename:
                continue
            filename = secure_filename(video.filename)
            video_path = os.path.join(UPLOAD_FOLDER, filename)
            video.save(video_path)
            video_urls.append(f'/uploads/{filename}')

        photo_urls_json = json.dumps(photo_urls)
        video_urls_json = json.dumps(video_urls)

        # mockup: ไม่เช็ค database จริง
        return jsonify({
            "post_id": id,
            "Title": Title,
            "content": content,
            "ProductName": ProductName,
            "CategoryID": CategoryID,
            "video_urls": video_urls,
            "photo_urls": photo_urls
        }), 200

    except Exception as error:
        print("Internal server error:", str(error))
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005) 