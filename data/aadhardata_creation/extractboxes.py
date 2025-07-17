from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# ✅ Custom draw_ocr function (since paddleocr.utils.visual is not available via pip)
def draw_ocr(img, boxes, txts, scores=None, font_path='arial.ttf'):
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, 20)
    except:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        draw.polygon(box, outline='red')
        txt = txts[i]
        if scores:
            txt += f' ({scores[i]:.2f})'
        draw.text(box[0], txt, fill='blue', font=font)
    return img

# 🔧 Load OCR model (make sure use_gpu=True)
ocr = PaddleOCR(lang='en', use_textline_orientation=True, det_db_box_thresh=0.3)

# 📷 Path to image to extract
img_path = "template.png"  # Change this to your Aadhaar image

# 🧠 Run OCR
result = ocr.ocr(img_path, cls=True)

# 📦 Extract result
boxes = [line[0] for line in result[0]]
txts = [line[1][0] for line in result[0]]
scores = [line[1][1] for line in result[0]]

# 🖍️ Draw OCR result
image = Image.open(img_path).convert("RGB")
font_path = "PingFang.ttf"  # or "arial.ttf"
im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)

# 💾 Save output
output_path = "ocr_result.jpg"
im_show.save(output_path)
print(f"OCR result saved to {output_path}")
