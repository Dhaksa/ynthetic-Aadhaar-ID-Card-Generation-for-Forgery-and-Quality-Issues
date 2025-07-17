import os
import re
import subprocess
import tempfile
import gradio as gr
from cvprocessor import main as edit_aadhar
from aiprocessor import generate_aadhar_image
from partialgenprocessor import create_partial_id, RedactionOption
from inpaintprocessor import flux_inpaint_ui

# --- OCR Field Detection Utility ---
def get_field_bbox(image_path, target_field):
    import easyocr
    import numpy as np
    import cv2

    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image_path)
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    for bbox, text, prob in results:
        text_lower = text.lower()
        if target_field == "aadhar_number" and re.search(r'\d{4}\s*\d{4}\s*\d{4}', text_lower):
            # bbox: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            x_coords = [pt[0] for pt in bbox]
            y_coords = [pt[1] for pt in bbox]
            return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)], width, height
        elif target_field == "name" and any(k in text_lower for k in ["name", "mohd", "sharukh"]):
            x_coords = [pt[0] for pt in bbox]
            y_coords = [pt[1] for pt in bbox]
            return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)], width, height
        elif target_field == "dob" and re.search(r'\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}', text_lower):
            x_coords = [pt[0] for pt in bbox]
            y_coords = [pt[1] for pt in bbox]
            return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)], width, height
    return None, width, height

def ensure_temp_dir():
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir

# --- Blender Occlusion Integration ---
def run_blender_occlusion(
    img_path,
    render_path,
    object_type,
    field,
    field_bbox,
    img_width,
    img_height,
    blender_executable,
    occlusion_script,
    coin_model, coin_texture,
    pen_model, pen_texture,
    pencil_model,pencil_texture,
   
):
    cmd = [
        blender_executable, "--background",
        "--python", occlusion_script,
        "--",
        "--img_path", img_path,
        "--object_type", object_type,
        "--field", field,
        "--render_path", render_path,
        "--field_bbox", str(field_bbox[0]), str(field_bbox[1]), str(field_bbox[2]), str(field_bbox[3]),
        "--img_width", str(img_width),
        "--img_height", str(img_height),
        "--coin_model", coin_model,
        "--coin_texture", coin_texture,
        "--pen_model", pen_model,
        "--pen_texture", pen_texture,
        "--pencil_model", pencil_model,
        "--pencil_texture", pencil_texture
    ]

    subprocess.run(cmd, check=True)
    return render_path


def extract_details_from_prompt(prompt):
    name = None
    dob = None
    aadhar_number = None
    vid = None
    name_patterns = [r'(?:name|called)\s+(?:is\s+)?([A-Za-z\s]+?)(?=\s+(?:and|with|dob|aadhar|vid|$))']
    dob_patterns = [r'(?:dob|date of birth|born on)\s+(?:is\s+)?(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})']
    aadhar_patterns = [r'(?:aadhar(?: number)?|id)\s+(?:is\s+)?(\d{4}\s*\d{4}\s*\d{4})']
    vid_patterns = [r'(?:vid|virtual id)\s+(?:is\s+)?(\d{4}\s*\d{4}\s*\d{4}\s*\d{4})']
    for pattern in name_patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            name = match.group(1).strip().title()
    for pattern in dob_patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            dob = match.group(1).replace("-", "/").replace(".", "/")
    for pattern in aadhar_patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            aadhar_number = re.sub(r'\s+', ' ', match.group(1))
    for pattern in vid_patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            vid = re.sub(r'\s+', ' ', match.group(1))
    return name, dob, aadhar_number, vid

def process_aadhar_card(input_image, redaction_option=None, apply_blur=False, output_path=None):
    try:
        if redaction_option == "None":
            redaction_opt = RedactionOption.NONE
        elif redaction_option == "Name":
            redaction_opt = RedactionOption.NAME_ONLY
        elif redaction_option == "Aadhaar Number":
            redaction_opt = RedactionOption.AADHAR_ONLY
        else:
            return "Invalid redaction option.", None
        if output_path is None:
            temp_dir = ensure_temp_dir()
            base_name = os.path.splitext(os.path.basename(input_image))[0]
            output_path = os.path.join(temp_dir, f"{base_name}_processed.jpg")
        result_path = create_partial_id(image_path=input_image,redaction_option=redaction_opt,apply_blur_effect=apply_blur,output_path=output_path)
        if result_path and os.path.exists(result_path):
            return "Aadhaar card processed successfully!", result_path
        else:
            return "Failed to process Aadhaar card.", None
    except Exception as e:
        return f"Error: {str(e)}", None

def generate_aadhar_card(prompt, generation_method="cv", apply_blur=False):
    name, dob, aadhar_number, vid = extract_details_from_prompt(prompt)
    if not any([name, dob, aadhar_number]):
        return "Could not extract any valid information. Please provide name, DOB, or Aadhar number.", None
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_image_path = os.path.join(project_root, "template.png")
    temp_dir = ensure_temp_dir()
    output_image_path = os.path.join(temp_dir, "generated_aadhar.jpg")
    try:
        if generation_method == "ai":
            ai_prompt = f"Aadhar card for {name or 'Random User'}"
            if dob:
                ai_prompt += f", born on {dob}"
            if aadhar_number:
                ai_prompt += f", Aadhar number {aadhar_number}"
            if vid:
                ai_prompt += f", VID {vid}"
            result_path = generate_aadhar_image(ai_prompt, output_path=output_image_path, apply_blur_effect=apply_blur)
        else:            
            result_path = edit_aadhar(
                image_path=input_image_path,
                new_name=name,
                new_dob=dob,
                new_aadhar=aadhar_number,
                new_vid=vid,
                output_path=output_image_path,
                apply_blur_effect=apply_blur
            )
        if os.path.exists(result_path):
            return "Aadhaar card generated successfully!", result_path
        else:
            return "Failed to generate Aadhaar card.", None
    except Exception as e:
        return f"Error: {str(e)}", None

# --- Gradio UI ---
def setup_gradio_ui():
    def update_input_accessibility(partial):
        return (
            gr.Textbox(interactive=not partial),  
            gr.Radio(interactive=not partial),    
            gr.Radio(visible=partial),            
            gr.Image(visible=partial)            
        )

    with gr.Blocks(title="HYPERGEN") as demo:
        with gr.Tabs():
            with gr.TabItem("Aadhaar Generator"):
                gr.Markdown("## HYPERGEN - Aadhaar Card Generator")
                gr.Markdown("Enter a prompt like: `Create aadhar card with name Jhon Doe and DOB 20/09/2003 and Aadhar number 1234 5678 1234`")

                with gr.Row():
                    with gr.Column(scale=2):
                        prompt_input = gr.Textbox(label="Prompt", placeholder="Describe the Aadhaar details...")
                        generation_method = gr.Radio(["CV-based", "AI-based"], label="Generation Method", value="CV-based")

                        with gr.Group(visible=True) as cv_options:
                            partial_id = gr.Checkbox(label="Process Existing ID", value=False)
                            upload_image = gr.Image(label="Upload Existing ID", type="filepath", visible=False)
                            redaction_options = gr.Radio(["None", "Name", "Aadhaar Number"], label="Select Fields to Redact", value="None", visible=False)
                            apply_blur = gr.Checkbox(label="Apply Blur Effect", value=False)

                            # --- Occlusion Controls ---
                            apply_occlusion = gr.Checkbox(label="Apply Physical Occlusion", value=False)
                            occlude_field = gr.Dropdown(["aadhar_number", "dob"], label="Field to Occlude", visible=False)
                            occlude_object = gr.Dropdown(["coin", "pen","pencil"], label="Object", visible=False)

                            apply_occlusion.change(
                                fn=lambda show: (gr.update(visible=show), gr.update(visible=show)),
                                inputs=[apply_occlusion],
                                outputs=[occlude_field, occlude_object]
                            )

                        generate_btn = gr.Button("Generate/Process Aadhaar Card")

                    with gr.Column(scale=3):
                        output_image = gr.Image(label="Result")
                        status_text = gr.Textbox(label="Status", interactive=False)

                def process_input(prompt, method, partial, upload_img, redact_opt, blur, occlude, field, obj):
                    # Step 1: Generate/process card
                    if method == "CV-based" and partial:
                        if upload_img is None:
                            return "Please upload an image first.", None
                        status, result_path = process_aadhar_card(upload_img, redact_opt, blur)
                    else:
                        gen_method = "cv" if method == "CV-based" else "ai"
                        status, result_path = generate_aadhar_card(prompt, gen_method, blur)

            
                    if result_path and occlude and field and obj:
                        
                        bbox, img_width, img_height = get_field_bbox(result_path, field)
                        if bbox is None:
                            return f"Could not find bounding box for field '{field}'.", None
                        temp_dir = ensure_temp_dir()
                        occluded_path = os.path.join(temp_dir, "occluded_render.png")
                        try:
                            run_blender_occlusion(
                                img_path=result_path,
                                render_path=occluded_path,
                                object_type=obj,
                                field=field,
                                field_bbox=bbox,
                                img_width=img_width,
                                img_height=img_height,
                                blender_executable="D:/Blender/blender.exe", 
                                occlusion_script="C:/Users/dhaks/Downloads/Hyperverge/src/occlude_render.py",  
                                coin_model="C:/Users/dhaks/Downloads/Hyperverge/3d-models/indian-coin/source/COIN.fbx",
                                coin_texture="C:/Users/dhaks/Downloads/Hyperverge/3d-models/indian-coin/textures/COIN.png",
                                pen_model="C:/Users/dhaks/Downloads/Hyperverge/3d-models/the-pen/source/Pen_Low.fbx",
                                pen_texture="C:/Users/dhaks/Downloads/Hyperverge/3d-models/the-pen/textures/Ruchka_Normal_DirectX.png",
                                pencil_model="C:/Users/dhaks/Downloads/Hyperverge/3d-models/pencil/source/Pencil.fbx",  
                                pencil_texture="C:/Users/dhaks/Downloads/Hyperverge/3d-models/pencil/textures/Pencil_normal.png"  
                                
                            )
                            status = f"Occlusion render successful with {obj} on {field}."
                            result_path = occluded_path
                        except Exception as e:
                            return f"Occlusion render failed: {e}", None

                    return status, result_path

                generation_method.change(fn=lambda x: x == "CV-based", inputs=[generation_method], outputs=[cv_options])
                partial_id.change(
                    fn=update_input_accessibility, 
                    inputs=[partial_id], 
                    outputs=[prompt_input, generation_method, redaction_options, upload_image]
                )

                generate_btn.click(
                    fn=process_input,
                    inputs=[
                        prompt_input, generation_method, partial_id,
                        upload_image, redaction_options, apply_blur,
                        apply_occlusion, occlude_field, occlude_object
                    ],
                    outputs=[status_text, output_image]
                )

            with gr.TabItem("Aadhar Inpainting"):
                flux_inpaint_ui()

    return demo

demo = setup_gradio_ui()

if __name__ == "__main__":
    ensure_temp_dir()
    demo.launch()
