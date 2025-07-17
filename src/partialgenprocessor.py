import easyocr
import cv2
from PIL import Image, ImageFilter
import numpy as np
import os
import re
from enum import Enum

def ensure_temp_dir():
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir

class RedactionOption(Enum):
    NAME_ONLY = "name"
    AADHAR_ONLY = "aadhar"
    NONE = "none"

def load_image(image_path):
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    image_height, image_width = image_cv.shape[:2]
    print(f"Image loaded: {image_path}, Dimensions: {image_width}x{image_height}")
    return image_cv, image_height, image_width

def initialize_ocr():
    return easyocr.Reader(['en'], gpu=False)

def detect_name_field(results, image_height):
    """Detect name field in OCR results"""
    name_bbox = None
    name_text = None
    
    for (bbox, text, prob) in results:
        y_top = bbox[0][1]
        
        is_proper_name = (
            text.replace(" ", "").isalpha() and
            len(text.split()) >= 1 and  
            prob > 0.5 and  
            y_top < image_height * 0.7 and  
            "government of india" not in text.lower() and
            len(text) > 3 
        )
        
        if is_proper_name:
            name_bbox = bbox
            name_text = text
            print(f"Name Field Detected: {text}")
            break
    
    if name_bbox is None:
        results_sorted = sorted(results, key=lambda x: x[0][0][1])
        for (bbox, text, prob) in results_sorted[:5]:
            if len(text) > 3 and prob > 0.4:
                name_bbox = bbox
                name_text = text
                print(f"Name Field (relaxed): {text}")
                break
    
    return name_bbox, name_text


def detect_aadhar_number(results):
    aadhar_bbox = None
    aadhar_text = None
    
    for (bbox, text, prob) in results:
        aadhar_pattern = r'\d{4}\s*\d{4}\s*\d{4}'
        matches = re.findall(aadhar_pattern, text)
        
        if matches or (len(text.replace(" ", "")) >= 10 and text.replace(" ", "").isdigit()):
            aadhar_bbox = bbox
            aadhar_text = text
            print(f"Aadhaar Number detected: {text}")
            break
    
    return aadhar_bbox, aadhar_text


def normalize_bbox(bbox):
    if isinstance(bbox[0], list):
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    else:
        return bbox


def expand_bbox(bbox, image_width, image_height, margin_ratio=0.08):
    x1, y1, x2, y2 = normalize_bbox(bbox)
    
    margin_x = int(image_width * margin_ratio)
    margin_y = int(image_height * margin_ratio)
    
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(image_width, x2 + margin_x)
    y2 = min(image_height, y2 + margin_y)
    
    return [x1, y1, x2, y2]


def create_partial_id_with_options(image_cv, name_bbox, aadhar_bbox, redaction_option, output_path, apply_blur_effect=False):
    image_height, image_width = image_cv.shape[:2]
    cropped_image = None
    
    if redaction_option == RedactionOption.NAME_ONLY:
        if name_bbox is not None:
            expanded_bbox = expand_bbox(name_bbox, image_width, image_height)
            print(f"Excluding name field: {expanded_bbox}")
            
            crop_y_start = expanded_bbox[3] + 1
            if crop_y_start < image_height:
                cropped_image = image_cv[crop_y_start:image_height, 0:image_width]
            else:
                crop_y_end = expanded_bbox[1] - 1
                cropped_image = image_cv[0:max(crop_y_end, image_height//3), 0:image_width]
    
    elif redaction_option == RedactionOption.AADHAR_ONLY:
        if aadhar_bbox is not None:
            expanded_bbox = expand_bbox(aadhar_bbox, image_width, image_height)
            print(f"Excluding Aadhaar field: {expanded_bbox}")
            
            crop_y_end = expanded_bbox[1] -1
            if crop_y_end > 0:
                cropped_image = image_cv[0:crop_y_end, 0:image_width]
            else:
                crop_y_start = expanded_bbox[3] + 1
                cropped_image = image_cv[crop_y_start:image_height, 0:image_width]
    
    result_img = cropped_image if cropped_image is not None else image_cv

    if apply_blur_effect:
        pil_image = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        try:
            import bpy
            from cvprocessor import apply_blender_blur
            blurred_array = apply_blender_blur(result_img)
            result_img = blurred_array
        except ImportError:
            blurred_image = pil_image.filter(ImageFilter.GaussianBlur(radius=5))
            result_img = cv2.cvtColor(np.array(blurred_image), cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, result_img)
    print(f"Processed image saved to: {output_path}")
    return output_path

def create_partial_id(image_path, redaction_option=None, apply_blur_effect=False, output_path=None):
    try:
        image_cv, image_height, image_width = load_image(image_path)
        reader = initialize_ocr()
        
        results = reader.readtext(image_path)
        name_bbox, name_text = detect_name_field(results, image_height)
        aadhar_bbox, aadhar_text = detect_aadhar_number(results)
        
        print(f"Name detected: {'Yes' if name_bbox else 'No'} - {name_text if name_text else 'N/A'}")
        print(f"Aadhaar detected: {'Yes' if aadhar_bbox else 'No'} - {aadhar_text if aadhar_text else 'N/A'}")

        temp_dir = ensure_temp_dir()
        if output_path is None:        
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(temp_dir, f"{base_name}_redacted_{redaction_option.value}.jpg")
        elif not os.path.isabs(output_path):
            output_path = os.path.join(temp_dir, output_path)
            
        result_path = create_partial_id_with_options(image_cv, name_bbox, aadhar_bbox, redaction_option, output_path, apply_blur_effect)
        
        return result_path
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None
    
    