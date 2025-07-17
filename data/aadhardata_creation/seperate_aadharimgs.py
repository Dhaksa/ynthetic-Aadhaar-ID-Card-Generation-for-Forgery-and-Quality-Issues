import os
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import shutil

IMAGE_DIR = "images"
AADHAAR_PHOTO_DIR = "aadhaar_photo_candidates"
NON_AADHAAR_DIR = "non_aadhaar_photos"

os.makedirs(AADHAAR_PHOTO_DIR, exist_ok=True)
os.makedirs(NON_AADHAAR_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

def is_suitable_for_aadhaar(image_path):
    try:
        image_path = os.path.abspath(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": "Is the photo suitable for Aadhaar card? The person should be facing straight, with a neutral expression, and a professional pose. Reply with only `true` or `false`."}
                ]
            }
        ]

        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, return_tensors="pt").to(DEVICE)

        generated_ids = model.generate(**inputs, max_new_tokens=5)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip().lower()

        return output_text == "true"
    except Exception as e:
        print(f"[ERROR] Failed to process {image_path}: {e}")
        return False

for filename in tqdm(os.listdir(IMAGE_DIR), desc="Classifying Aadhaar-suitable photos"):
    filepath = os.path.join(IMAGE_DIR, filename)
    if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
        continue

    if is_suitable_for_aadhaar(filepath):
        shutil.copy(filepath, os.path.join(AADHAAR_PHOTO_DIR, filename))
    else:
        shutil.copy(filepath, os.path.join(NON_AADHAAR_DIR, filename))

print(f"\n- {AADHAAR_PHOTO_DIR}/ (suitable for Aadhaar photo)")
print(f"- {NON_AADHAAR_DIR}/ (not suitable)")
