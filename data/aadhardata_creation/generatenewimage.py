import cv2
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import csv

# ---------- CONFIG ----------
TEMPLATE_PATH = "template.png"
CSV_PATH = "data.csv"
FONT_PATH = "arial.ttf"
OUTPUT_DIR = "generated_output"
# ----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_bbox(data, target, group_size=1):
    for i in range(len(data["text"]) - (group_size - 1)):
        group = data["text"][i:i + group_size]
        joined = "".join(group).replace(" ", "")
        if joined == target.replace(" ", ""):
            x1 = data["left"][i]
            y1 = min(data["top"][i + j] for j in range(group_size))
            x2 = data["left"][i + group_size - 1] + data["width"][i + group_size - 1]
            y2 = max(data["top"][i + j] + data["height"][i + j] for j in range(group_size))
            return [x1, y1, x2, y2]
    return None

def get_name_bbox(data, keywords):
    name_coords = None
    for i, word in enumerate(data["text"]):
        if any(k.lower() in word.lower() for k in keywords):
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            if name_coords is None:
                name_coords = [x, y, x + w, y + h]
            else:
                name_coords[0] = min(name_coords[0], x)
                name_coords[1] = min(name_coords[1], y)
                name_coords[2] = max(name_coords[2], x + w)
                name_coords[3] = max(name_coords[3], y + h)
    return name_coords

def replace_text(draw, coords, new_text, font_path, adjust_y=0):
    if coords:
        x1, y1, x2, y2 = coords
        width, height = x2 - x1, y2 - y1
        font_size = height + 2

        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()

        draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
        _, _, text_width, text_height = font.getbbox(new_text)
        y_aligned = y1 + (height - text_height) // 2 + adjust_y
        draw.text((x1, y_aligned), new_text, font=font, fill=(0, 0, 0))
    else:
        print(f"Warning: Could not find bbox for {new_text}")

def process_row(template_img, data, row, output_path):
    pil_img = Image.fromarray(template_img.copy())
    draw = ImageDraw.Draw(pil_img)

    name_coords = get_name_bbox(data, ["Mohd", "Sharukh"])  # Modify keywords from template
    dob_coords = find_bbox(data, "13/03/1996")
    uid_coords = find_bbox(data, "455158937035", group_size=3)
    vid_coords = find_bbox(data, "9163912924645515", group_size=4)

    replace_text(draw, name_coords, row["name"], FONT_PATH, adjust_y=-2)
    replace_text(draw, dob_coords, row["dob"], FONT_PATH, adjust_y=-2)
    replace_text(draw, uid_coords, row["aadhar number"], FONT_PATH)
    replace_text(draw, vid_coords, row["vid"], FONT_PATH, adjust_y=-2)

    output_image = np.array(pil_img)
    cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

def main():
    if not os.path.exists(TEMPLATE_PATH):
        raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")

    template = cv2.imread(TEMPLATE_PATH)
    rgb_template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb_template, output_type=Output.DICT)

    with open(CSV_PATH, newline='', encoding='utf-8-sig', errors='replace') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            output_file = os.path.join(OUTPUT_DIR, f"output_{i+1}_{row['name'].replace(' ', '_')}.jpg")
            print(f"Generating: {output_file}")
            process_row(rgb_template, data, row, output_file)

if __name__ == "__main__":
    main()
