# occlusion_blender.py

import bpy
import sys
import argparse
import random
import os

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", required=True)
    parser.add_argument("--object_type", required=True, choices=["coin", "pen", "pencil"])
    parser.add_argument("--field", required=True, choices=["aadhar_number", "name", "dob"])
    parser.add_argument("--render_path", required=True)
    parser.add_argument("--field_bbox", nargs=4, type=float, required=True, help="Bounding box x1 y1 x2 y2 for field occlusion")
    parser.add_argument("--img_width", type=int, required=True)
    parser.add_argument("--img_height", type=int, required=True)
    parser.add_argument("--coin_model", required=True)
    parser.add_argument("--coin_texture", required=True)
    parser.add_argument("--pen_model", required=True)
    parser.add_argument("--pen_texture", required=True)
    parser.add_argument("--pencil_model", required=True)
    parser.add_argument("--pencil_texture", required=True)
    return parser.parse_args(argv)

def pixel_to_blender_coords(bbox, image_width, image_height, plane_size=2):
    x1, y1, x2, y2 = bbox
    center_x_px = (x1 + x2) / 2
    center_y_px = (y1 + y2) / 2
    norm_x = center_x_px / image_width
    norm_y = center_y_px / image_height
    blender_x = (norm_x - 0.5) * plane_size
    blender_y = (0.5 - norm_y) * plane_size  # Y axis flipped
    return blender_x, blender_y

def add_object(object_type, args, obj_x, obj_y):
    if object_type == "coin":
        fbx_path = args.coin_model
        texture_path = args.coin_texture
        scale_range = (0.10, 0.25)
    elif object_type == "pen":
        fbx_path = args.pen_model
        texture_path = args.pen_texture
        scale_range = (0.4, 0.6)
    elif object_type == "pencil":
        fbx_path = args.pencil_model
        texture_path = args.pencil_texture
        scale_range = (0.4, 0.6)

    bpy.ops.import_scene.fbx(filepath=fbx_path)
    imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if not imported_objects:
        print("No mesh found in FBX import.")
        return None
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in imported_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported_objects[0]
    bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.name = object_type.capitalize()

    rand_scale = random.uniform(*scale_range)
    obj.scale = (rand_scale, rand_scale, rand_scale)
    obj.location = (obj_x, obj_y, 0.015)
    obj.rotation_euler = (0, 0, 0)

    mat = bpy.data.materials.new(name=f"{object_type.capitalize()}Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    tex_image = nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(texture_path)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return obj

def main():
    args = parse_args()
    # Clean scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Add table
    bpy.ops.mesh.primitive_plane_add(size=5, location=(0, 0, -0.01))
    table = bpy.context.active_object
    table.name = "Surface"
    mat = bpy.data.materials.new("TableMat")
    mat.use_nodes = True
    table.data.materials.append(mat)
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.2, 0.2, 0.2, 1)
    table.cycles.is_shadow_catcher = True

    # Add card
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    card = bpy.context.active_object
    card.name = "Aadhar_Card"
    card_mat = bpy.data.materials.new("AadharCardMaterial")
    card_mat.use_nodes = True
    nodes = card_mat.node_tree.nodes
    nodes.clear()
    tex_image = nodes.new("ShaderNodeTexImage")
    tex_image.image = bpy.data.images.load(args.img_path)
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    output = nodes.new("ShaderNodeOutputMaterial")
    card_mat.node_tree.links.new(tex_image.outputs["Color"], diffuse.inputs["Color"])
    card_mat.node_tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    card.data.materials.append(card_mat)

    # Camera
    bpy.ops.object.camera_add(location=(0, 0, 3), rotation=(0, 0, 0))
    cam = bpy.context.active_object
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 2.5
    bpy.context.scene.camera = cam

    # Light
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 5))
    light = bpy.context.active_object
    light.data.energy = 350
    light.data.size = 3
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0.1

    # Calculate object position from bbox
    obj_x, obj_y = pixel_to_blender_coords(
        args.field_bbox, args.img_width, args.img_height, plane_size=2
    )

    # Add object at calculated position
    obj = add_object(args.object_type, args, obj_x, obj_y)
    if obj is None:
        print("Failed to add object.")
        return

    # Render settings
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.render.resolution_x = args.img_width
    scene.render.resolution_y = args.img_height
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = args.render_path
    scene.cycles.samples = 256

    bpy.ops.render.render(write_still=True)
    print("Rendered with {} at: {}".format(args.object_type, args.render_path))

if __name__ == "__main__":
    main()


