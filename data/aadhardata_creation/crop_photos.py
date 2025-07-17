import cv2
import os

# Input and output directories
input_folder = "images"  # folder with original Aadhaar card images
output_folder = "cropped_photos"  # folder to store cropped profile photos

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Process each image in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(input_folder, filename)
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.6 < aspect_ratio < 1.2 and 100 < w < 400 and x < img.shape[1] // 2:
                candidates.append((x, y, w, h))

        # Choose the topmost valid candidate
        if candidates:
            x, y, w, h = sorted(candidates, key=lambda b: b[1])[0]
            photo_crop = img[y:y+h, x:x+w]

            # Save with same name in output folder
            output_path = os.path.join(output_folder, f"cropped_{filename}")
            cv2.imwrite(output_path, photo_crop)
            print(f"[✓] Cropped photo saved: {output_path}")
        else:
            print(f"[X] No suitable photo found in: {filename}")
