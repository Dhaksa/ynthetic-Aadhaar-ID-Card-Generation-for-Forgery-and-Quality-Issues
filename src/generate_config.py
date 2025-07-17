import cv2
import easyocr
import json
import os
from partialgenprocessor import detect_name_field, detect_aadhar_number, expand_bbox

def generate_config(
    image_path,
    selected_field="name",     # 'name' or 'aadhar'
    object_type="coin",        # 'coin', 'pen', etc.
    output_path="output/render.png"
):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    h, w = image.shape[:2]

    # Run OCR
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image)

    # Choose field to occlude
    if selected_field == "name":
        bbox, text = detect_name_field(results, h)
    elif selected_field == "aadhar":
        bbox, text = detect_aadhar_number(results)
    else:
        raise ValueError("Invalid field selected. Choose either 'name' or 'aadhar'.")

    if bbox is None:
        raise Exception(f"Could not detect the {selected_field} field.")

    # Expand the bounding box slightly
    target_coords = expand_bbox(bbox, w, h, margin_ratio=0.05)

    # Prepare config data
    config = {
        "img_path": os.path.abspath(image_path),
        "object": object_type,
        "target_coords": target_coords,
        "render_path": os.path.abspath(output_path)
    }

    # Write to config.json
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("config.json written successfully!")
    print(json.dumps(config, indent=2))

# Example usage
if __name__ == "__main__":
    generate_config(
        image_path="aadhaar_sample.jpg",
        selected_field="aadhar",      # or "name"
        object_type="pen",            # or "coin"
        output_path="output/occluded_output.png"
    )
