import torch
import gradio as gr
import os
from PIL import Image
from diffusers import FluxFillPipeline

pipe = None

def ensure_temp_dir():
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir

def load_model():
    global pipe
    if pipe is None:
        pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev",torch_dtype=torch.float16).to("cuda")
        return "Model loaded successfully."
    else:
        return "Model is already loaded."

def calculate_optimal_dimensions(image):
    original_width, original_height = image.size
    aspect = original_width / original_height
    FIXED = 1024

    if aspect > 1:
        width, height = FIXED, round(FIXED / aspect)
    else:
        height, width = FIXED, round(FIXED * aspect)

    width = max((width // 8) * 8, 576)
    height = max((height // 8) * 8, 576)

    return width, height

def inpaint_with_mask(img_data, prompt="", num_inference_steps=30, guidance_scale=80):
    if pipe is None:
        return "⚠️ Load the model first."
    if img_data is None:
        return None

    base_image = Image.fromarray(img_data["image"]).convert("RGB")
    mask = Image.fromarray(img_data["mask"]).convert("L")
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

    temp_dir = ensure_temp_dir()
    output_path = os.path.join(temp_dir, "inpainted_result.png")
    result.save(output_path)
    print(f"Inpainted image saved to: {output_path}")

    return result.convert("RGBA")

def flux_inpaint_ui():
    with gr.Blocks() as flux_tab:
        gr.Markdown("## Aadhar Inpainting - Modify Existing Aadhar Card")
        gr.Markdown("Enter a prompt like: `Replace the existing name with 'VISHWA', preserving the exact same font, size, style, and alignment. Ensure the new text seamlessly matches the surrounding background and lighting so that it looks natural and unaltered.`")

        with gr.Row():
            with gr.Column(scale=1):
                load_btn = gr.Button("Load Model")
            with gr.Column(scale=1):
                load_status = gr.Textbox(label="", interactive=False)
        load_btn.click(fn=load_model, outputs=load_status)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.ImageMask(label="Upload and draw mask", type="numpy", height=400)
            with gr.Column(scale=1):
                output_img = gr.Image(label="Inpainted Output", height=400)

        prompt = gr.Textbox(label="Prompt", value="")
        steps = gr.Number(label="Inference Steps", value=30)
        scale = gr.Number(label="Guidance Scale", value=80)

        inpaint_btn = gr.Button("Run Inpainting")
        inpaint_btn.click(fn=inpaint_with_mask, inputs=[image_input, prompt, steps, scale], outputs=output_img)

    return flux_tab
