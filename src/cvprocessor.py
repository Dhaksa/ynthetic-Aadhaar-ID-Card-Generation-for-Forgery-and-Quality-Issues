import cv2
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import bpy
import tempfile

def ensure_temp_dir():
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir

def get_best_fit_font(text, target_width, target_height, font_path="arial.ttf"):
    font_size = target_height
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
        return font

    while font_size > 1:
        bbox = font.getbbox(text)
        if bbox[2] <= target_width and bbox[3] <= target_height:
            break
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)

    return font

def find_bbox(data, target, group_size=1):
    for i in range(len(data["text"]) - (group_size - 1)):
        group = data["text"][i:i + group_size]
        joined = "".join(group).replace(" ", "")
        if joined == target.replace(" ", ""):
            x1 = data["left"][i]
            y1 = min(data["top"][i + j] for j in range(group_size))
            x2 = data["left"][i + group_size - 1] + data["width"][i + group_size - 1]
            y2 = max(data["top"][i + j] + data["height"][i + j] for j in range(group_size))
            return [x1, y1, x2, y2]
    return None

def apply_blender_blur(image_array, blur_strength=10):
    """
    Attempts to blur using Blender compositor. If Blender isn't available, falls back to OpenCV Gaussian blur.
    Accepts and returns NumPy image arrays in RGB format.
    """
    try:
        import bpy
        import tempfile
        import os
 
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_in, \
             tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_out:
            temp_input = temp_in.name
            temp_output = temp_out.name
 
        # Save input image
        cv2.imwrite(temp_input, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
 
        # Set up Blender compositor
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        tree.nodes.clear()
 
        image_node = tree.nodes.new('CompositorNodeImage')
        blur_node = tree.nodes.new('CompositorNodeBlur')
        composite_node = tree.nodes.new('CompositorNodeComposite')
        viewer_node = tree.nodes.new('CompositorNodeViewer')
 
        image = bpy.data.images.load(temp_input)
        image_node.image = image
 
        blur_node.filter_type = 'GAUSS'
        blur_node.size_x = blur_strength
        blur_node.size_y = blur_strength
 
        links = tree.links
        links.new(image_node.outputs['Image'], blur_node.inputs['Image'])
        links.new(blur_node.outputs['Image'], composite_node.inputs['Image'])
        links.new(blur_node.outputs['Image'], viewer_node.inputs['Image'])
 
        bpy.context.scene.render.resolution_x = image.size[0]
        bpy.context.scene.render.resolution_y = image.size[1]
        bpy.context.scene.render.filepath = temp_output
        bpy.ops.render.render(use_viewport=False, write_still=True)
 
        blurred_image = cv2.imread(temp_output)
        blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
 
        os.unlink(temp_input)
        os.unlink(temp_output)
 
        return blurred_image
 
    except Exception as e:
        print(f"[Fallback] Blender blur failed: {e}")
        # Fallback using OpenCV Gaussian Blur
        return cv2.GaussianBlur(image_array, (blur_strength | 1, blur_strength | 1), 0)

def main(image_path, new_name=None, new_dob=None, new_aadhar=None, new_vid=None,
         output_path=None, apply_blur_effect=False):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Template image not found at {image_path}")

    # Ensure temp directory exists and set default output path
    temp_dir = ensure_temp_dir()
    if output_path is None:
        output_path = os.path.join(temp_dir, "cv_generated_aadhar.jpg")
    elif not os.path.isabs(output_path):
        output_path = os.path.join(temp_dir, output_path)

    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb_image, output_type=Output.DICT)

    pil_img = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_img)

    font_path = "arial.ttf"

    target_name_keywords = ["Mohd", "Sharukh"]
    target_dob = "13/03/1996"
    target_uid = "455158937035"
    target_vid = "9163912924645515"

    name_coords = None
    for i, word in enumerate(data["text"]):
        if any(k.lower() in word.lower() for k in target_name_keywords):
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            if name_coords is None:
                name_coords = [x, y, x + w, y + h]
            else:
                name_coords[0] = min(name_coords[0], x)
                name_coords[1] = min(name_coords[1], y)
                name_coords[2] = max(name_coords[2], x + w)
                name_coords[3] = max(name_coords[3], y + h)

    uid_coords = find_bbox(data, target_uid, group_size=3)
    dob_coords = find_bbox(data, target_dob, group_size=1)
    vid_coords = find_bbox(data, target_vid, group_size=4)

    def replace_text(coords, new_text, adjust_y=0, size_boost=0):
        if coords:
            x1, y1, x2, y2 = coords
            width, height = x2 - x1, y2 - y1

            adjusted_height = height + size_boost

            draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
            font = get_best_fit_font(new_text, width, adjusted_height + 4, font_path)
            text_width, text_height = font.getbbox(new_text)[2], font.getbbox(new_text)[3]
            
            y_aligned = y1 + (adjusted_height - text_height) // 2 + adjust_y
            draw.text((x1, y_aligned), new_text, font=font, fill=(0, 0, 0))
        else:
            print(f"Could not find bounding box for: {new_text}")


    if new_name:
        replace_text(name_coords, new_name,size_boost=1, adjust_y=-2)
    if new_dob:
        replace_text(dob_coords, new_dob, adjust_y=-2)
    if new_aadhar:
        replace_text(uid_coords, new_aadhar)
    if new_vid:
        replace_text(vid_coords, new_vid, adjust_y=-2)

    result_img = np.array(pil_img)

    if apply_blur_effect:
        result_img = apply_blender_blur(result_img)

    cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    return output_path