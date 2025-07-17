import torch
import gradio as gr
from PIL import Image
from diffusers import FluxFillPipeline

pipe = None

def load_model():
    global pipe
    if pipe is None:
        pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev",torch_dtype=torch.float16).to("cuda")
        return "Model loaded successfully."
    else:
        return "Model is already loaded."

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


def inpaint_with_mask(img_data, prompt="", num_inference_steps=30, guidance_scale=80):
    if pipe is None:
        return "⚠️ Please load the model first by clicking the button above."

    if img_data is None:
        return None

    base_image = Image.fromarray(img_data["image"]).convert("RGB")
    mask_layer = img_data["mask"]
    mask = Image.fromarray(mask_layer).convert("L")

    width, height = calculate_optimal_dimensions(base_image)

    result = pipe(
        prompt=prompt,
        height=height,
        width=width,
        image=base_image,
        mask_image=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    return result.convert("RGBA")


with gr.Blocks() as demo:
    gr.Markdown("##FLUX Inpainting - Upload, Mask, and Inpaint")

    with gr.Row():
        load_btn = gr.Button("Load Model")
        load_status = gr.Textbox(label="", interactive=False)

    load_btn.click(fn=load_model, outputs=load_status)

    with gr.Row():
        image_input = gr.ImageMask(label="Upload and draw mask", type="numpy")
        output_img = gr.Image(label="Inpainted Output")

    prompt = gr.Textbox(label="Prompt", value="")
    steps = gr.Number(label="Inference Steps", value=28)
    scale = gr.Number(label="Guidance Scale", value=50)

    inpaint_btn = gr.Button("Run Inpainting")

    inpaint_btn.click(fn=inpaint_with_mask, inputs=[image_input, prompt, steps, scale], outputs=output_img)

demo.launch(debug=True, show_error=True)
