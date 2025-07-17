import torch
import gradio as gr
from diffusers import FluxFillPipeline

pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.float16).to("cuda")

def calculate_optimal_dimensions(image):
    original_width, original_height = image.size
    
    MIN_ASPECT_RATIO = 9 / 16
    MAX_ASPECT_RATIO = 16 / 9
    FIXED_DIMENSION = 1024

    original_aspect_ratio = original_width / original_height

    if original_aspect_ratio > 1:
        width = FIXED_DIMENSION
        height = round(FIXED_DIMENSION / original_aspect_ratio)
    else:
        height = FIXED_DIMENSION
        width = round(FIXED_DIMENSION * original_aspect_ratio)

    width = (width // 8) * 8
    height = (height // 8) * 8

    calculated_aspect_ratio = width / height
    if calculated_aspect_ratio > MAX_ASPECT_RATIO:
        width = (height * MAX_ASPECT_RATIO // 8) * 8
    elif calculated_aspect_ratio < MIN_ASPECT_RATIO:
        height = (width / MIN_ASPECT_RATIO // 8) * 8

    width = max(width, 576) if width == FIXED_DIMENSION else width
    height = max(height, 576) if height == FIXED_DIMENSION else height

    return width, height


def inpaint(image,mask,prompt="",num_inference_steps=28,guidance_scale=50,):
    image = image.convert("RGB")
    mask = mask.convert("L")
    width, height = calculate_optimal_dimensions(image)

    result = pipe(prompt=prompt,height= height,width= width,image= image,mask_image=mask,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,).images[0]
    result = result.convert("RGBA")
    return result


demo = gr.Interface(
    fn=inpaint,
    inputs=[
        gr.Image(label="image", type="pil"),
        gr.Image(label="mask", type="pil"),
        gr.Text(label="prompt"),
        gr.Number(value=40, label="num_inference_steps"),
        gr.Number(value=28, label="guidance_scale"),
    ],
    outputs=["image"],
    api_name="inpaint"
)

demo.launch(debug=True,show_error=True)