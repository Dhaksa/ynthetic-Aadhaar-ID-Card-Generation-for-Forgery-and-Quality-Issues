import easyocr
import cv2
import numpy as np
import os
import re
import argparse
from enum import Enum


class RedactionOption(Enum):
    """Enumeration for redaction options"""
    NAME_ONLY = "name"
    AADHAR_ONLY = "aadhar"
    NONE = "none"


def load_image(image_path):
    """Load image and return image object with dimensions"""
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    image_height, image_width = image_cv.shape[:2]
    print(f"Image loaded: {image_path}, Dimensions: {image_width}x{image_height}")
    return image_cv, image_height, image_width


def initialize_ocr():
    """Initialize EasyOCR reader"""
    print("Initializing OCR reader...")
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
    """Detect Aadhaar number in OCR results"""
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
    """Convert bbox to standard format: [x1, y1, x2, y2]"""
    if isinstance(bbox[0], list):
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    else:
        return bbox


def expand_bbox(bbox, image_width, image_height, margin_ratio=0.08):
    """Expand bounding box by a margin to ensure complete field coverage"""
    x1, y1, x2, y2 = normalize_bbox(bbox)
    
    margin_x = int(image_width * margin_ratio)
    margin_y = int(image_height * margin_ratio)
    
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(image_width, x2 + margin_x)
    y2 = min(image_height, y2 + margin_y)
    
    return [x1, y1, x2, y2]


def create_partial_id_with_options(image_cv, name_bbox, aadhar_bbox, redaction_option, output_path):
    """Create partial ID based on redaction options"""
    image_height, image_width = image_cv.shape[:2]
    
    if redaction_option == RedactionOption.NAME_ONLY:
        if name_bbox is not None:
            expanded_bbox = expand_bbox(name_bbox, image_width, image_height)
            print(f"Excluding name field: {expanded_bbox}")
            
            crop_y_start = expanded_bbox[3] + 20
            if crop_y_start < image_height:
                cropped_image = image_cv[crop_y_start:image_height, 0:image_width]
            else:
                crop_y_end = expanded_bbox[1] - 20
                cropped_image = image_cv[0:max(crop_y_end, image_height//3), 0:image_width]
        else:
            print("Warning: Name field not detected, cannot exclude")
            cv2.imwrite(output_path, image_cv)
            return output_path
    
    elif redaction_option == RedactionOption.AADHAR_ONLY:
        if aadhar_bbox is not None:
            expanded_bbox = expand_bbox(aadhar_bbox, image_width, image_height)
            print(f"Excluding Aadhaar field: {expanded_bbox}")
            
            crop_y_end = expanded_bbox[1] - 20  
            if crop_y_end > 0:
                cropped_image = image_cv[0:crop_y_end, 0:image_width]
            else:
                crop_y_start = expanded_bbox[3] + 20
                cropped_image = image_cv[crop_y_start:image_height, 0:image_width]
        else:
            print("Warning: Aadhaar field not detected, cannot exclude")
            cv2.imwrite(output_path, image_cv)
            return output_path
    
    else:
        print("No redaction requested, saving original image")
        cv2.imwrite(output_path, image_cv)
        return output_path
    
    cv2.imwrite(output_path, cropped_image)
    print(f"Partial ID created with {redaction_option.value} redaction: {output_path}")
    print(f"Original: {image_width}x{image_height}, Cropped: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
    
    return output_path


def get_redaction_option_interactive():
    """Interactive function to get user's redaction preference"""
    print("\n=== Aadhaar Card Redaction Options ===")
    print("1. Redact Name Only")
    print("2. Redact Aadhaar Number Only") 
    print("3. No Redaction (Keep Original)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                return RedactionOption.NAME_ONLY
            elif choice == "2":
                return RedactionOption.AADHAR_ONLY
            elif choice == "3":
                return RedactionOption.NONE
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None


def create_partial_id(image_path, redaction_option=None, output_path=None):
    """Main function to create partial ID with flexible redaction options"""
    
    try:
        image_cv, image_height, image_width = load_image(image_path)
        reader = initialize_ocr()
        
        results = reader.readtext(image_path)
        name_bbox, name_text = detect_name_field(results, image_height)
        aadhar_bbox, aadhar_text = detect_aadhar_number(results)
        
        print(f"\n=== Detection Results ===")
        print(f"Name detected: {'Yes' if name_bbox else 'No'} - {name_text if name_text else 'N/A'}")
        print(f"Aadhaar detected: {'Yes' if aadhar_bbox else 'No'} - {aadhar_text if aadhar_text else 'N/A'}")
        
        if redaction_option is None:
            redaction_option = get_redaction_option_interactive()
            if redaction_option is None:
                return None
        
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{base_name}_redacted_{redaction_option.value}.jpg"
        
        result_path = create_partial_id_with_options(
            image_cv, name_bbox, aadhar_bbox, redaction_option, output_path
        )
        
        return result_path
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None


def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description="Redact sensitive information from Aadhaar cards")
    parser.add_argument("image_path", help="Path to the Aadhaar card image")
    parser.add_argument("-r", "--redact", choices=["name", "aadhar", "none"], 
                       help="Redaction option: name, aadhar, or none")
    parser.add_argument("-o", "--output", help="Output file path")
    
    args = parser.parse_args()
    
    redaction_map = {
        "name": RedactionOption.NAME_ONLY,
        "aadhar": RedactionOption.AADHAR_ONLY,
        "none": RedactionOption.NONE
    }
    
    redaction_option = redaction_map.get(args.redact) if args.redact else None
    
    result = create_partial_id(args.image_path, redaction_option, args.output)
    
    if result:
        print(f"\n✅ Success! Redacted image saved: {result}")
    else:
        print("\n❌ Failed to process image")


if __name__ == "__main__":
    main()

# Command line usage examples:
# python script.py aadhar_card.jpg --redact name
# python script.py aadhar_card.jpg --redact aadhar --output custom_output.jpg
# python script.py aadhar_card.jpg  # Interactive mode