from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch

# Load local image and mask
image = Image.open("template.png").convert("RGB")
mask = Image.open("templatemask.png").convert("RGB")

# Use tokenizer and text encoder (standard, non-quantized)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# Load inpainting pipeline WITHOUT revision param
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    torch_dtype=torch.float16,
    safety_checker=None,
)

# Memory optimization
pipe.enable_attention_slicing()
pipe.vae.to(torch.float32, "cpu")  # Move VAE to CPU to save VRAM
pipe.to("cuda")

# Run
output = pipe(
    prompt="Replace the existing name with 'VISHWA', preserving the exact same font, size, style, and alignment. Ensure the new text seamlessly matches the surrounding background and lighting so that it looks natural and unaltered.",
    image=image,
    mask_image=mask,
    guidance_scale=7.5,
    num_inference_steps=30,
).images[0]

output.save("3050_inpaint_output.png")
