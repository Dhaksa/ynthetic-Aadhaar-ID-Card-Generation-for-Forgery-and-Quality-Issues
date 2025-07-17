import cv2
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

# Helper: Fit font to bounding box
def get_best_fit_font(text, target_width, target_height, font_path):
    font_size = target_height
    font = ImageFont.truetype(font_path, font_size)
    while (font.getbbox(text)[2] > target_width or font.getbbox(text)[3] > target_height) and font_size > 1:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
    return font

# Helper: Find bounding box by matching full string (joining group of words)
def find_bbox(targets, group_size=1):
    for i in range(len(data["text"]) - (group_size - 1)):
        group = data["text"][i:i+group_size]
        joined = "".join(group).replace(" ", "")
        if joined in targets:
            x1 = data["left"][i]
            y1 = min(data["top"][i + j] for j in range(group_size))
            x2 = data["left"][i + group_size - 1] + data["width"][i + group_size - 1]
            y2 = max(data["top"][i + j] + data["height"][i + j] for j in range(group_size))
            return [x1, y1, x2, y2]
    return None

# Aadhaar card image path
image_path = "template.png"  # Update as needed
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
data = pytesseract.image_to_data(rgb_image, output_type=Output.DICT)

# Font path (bold Arial)
font_path = "C:/Windows/Fonts/arial.ttf"

# New values
target_name_keywords = ["Mohd", "Sharukh"]
new_name = "VISHWA"

target_uid = "455158937035"
new_uid = "1234 5678 9123"

target_dob = "13/03/1996"
new_dob = "17/10/2004"

target_vid = "9163912924645515"
new_vid = "1234 5678 9012 3456"

# ----- Locate bounding boxes -----
# Name
name_coords = None
for i, word in enumerate(data["text"]):
    if any(k in word for k in target_name_keywords):
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if name_coords is None:
            name_coords = [x, y, x + w, y + h]
        else:
            name_coords[0] = min(name_coords[0], x)
            name_coords[1] = min(name_coords[1], y)
            name_coords[2] = max(name_coords[2], x + w)
            name_coords[3] = max(name_coords[3], y + h)

uid_coords = find_bbox([target_uid], group_size=3)
dob_coords = find_bbox([target_dob], group_size=1)
vid_coords = find_bbox([target_vid], group_size=4)

# ----- Draw replacements -----
pil_img = Image.fromarray(rgb_image)
draw = ImageDraw.Draw(pil_img)

def replace_text(coords, new_text):
    if coords:
        x1, y1, x2, y2 = coords
        width, height = x2 - x1, y2 - y1
        draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
        font = get_best_fit_font(new_text, width, height + 2, font_path)
        text_width, text_height = font.getbbox(new_text)[2], font.getbbox(new_text)[3]
        y_aligned = y1 + (height - text_height) // 2
        draw.text((x1, y_aligned), new_text, font=font, fill=(0, 0, 0))
    else:
        print(f"Could not find bounding box for: {new_text}")

# Apply replacements
replace_text(name_coords, new_name)
replace_text(uid_coords, new_uid)
replace_text(dob_coords, new_dob)
replace_text(vid_coords, new_vid)

# ----- Save & Display -----
result_img = np.array(pil_img)
result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("aadhar_modified.png", result_bgr)

# Load original again for comparison
original_image = cv2.imread(image_path)
original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Side-by-side display
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].imshow(original_rgb)
axs[0].set_title("Original Aadhaar Card")
axs[0].axis("off")

axs[1].imshow(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
axs[1].set_title("Modified Aadhaar Card")
axs[1].axis("off")

plt.tight_layout()
plt.show()
