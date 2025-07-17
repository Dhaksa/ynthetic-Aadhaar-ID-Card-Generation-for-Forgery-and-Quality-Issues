import torch
import os
from diffusers import StableDiffusionPipeline
from PIL import Image

def ensure_temp_dir():
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir

def download_model():
    return StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32  )

def generate_aadhar_image(prompt, output_path=None, apply_blur_effect=False, blur_strength=2):
    try:
        temp_dir = ensure_temp_dir()
        if output_path is None:
            output_path = os.path.join(temp_dir, "ai_generated_aadhar.jpg")
        elif not os.path.isabs(output_path):
            output_path = os.path.join(temp_dir, output_path)
        
        enhanced_prompt = f"An official Indian Aadhar card, government ID document, with details, {prompt}, highly detailed, realistic"
        
        device = "cpu"
        print(f"Using device: {device}")
        
        pipe = download_model()
        pipe = pipe.to(device)
        
        print(f"Generating image with prompt: {enhanced_prompt}")
        with torch.no_grad():
            image = pipe(enhanced_prompt).images[0]
        
        if apply_blur_effect:
            print(f"Applying blur effect with strength {blur_strength}")
            image = image.filter(Image.FILTER.GaussianBlur(radius=blur_strength))
        
        image.save(output_path)
        print(f"AI-generated image saved as: {output_path}")
        
        return output_path
    
    except Exception as e:
        print(f"Error generating image with AI: {str(e)}")
        
        temp_dir = ensure_temp_dir()
        if output_path is None:
            output_path = os.path.join(temp_dir, "ai_generated_aadhar.jpg")
        elif not os.path.isabs(output_path):
            output_path = os.path.join(temp_dir, output_path)
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fallback_path = os.path.join(project_root, "data", "aadhar_card.png")
        if os.path.exists(fallback_path):
            print("Using fallback template image instead")
            fallback_image = Image.open(fallback_path)
            
            if apply_blur_effect:
                fallback_image = fallback_image.filter(Image.FILTER.GaussianBlur(radius=blur_strength))
                
            fallback_image.save(output_path)
            return output_path
        return None
