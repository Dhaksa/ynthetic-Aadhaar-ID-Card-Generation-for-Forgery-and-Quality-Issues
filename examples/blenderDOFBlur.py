import bpy

input_image_path = "J:/HyperGen/BlenderBlur/check.png"
depth_map_path = "J:/HyperGen/BlenderBlur/depth_mapcheck.png"
output_path = "J:/HyperGen/BlenderBlur/blurred_output.jpg"

scene = bpy.context.scene
scene.use_nodes = True
tree = scene.node_tree
tree.nodes.clear()

image_node = tree.nodes.new(type="CompositorNodeImage")
image_node.image = bpy.data.images.load(input_image_path)

depth_node = tree.nodes.new(type="CompositorNodeImage")
depth_node.image = bpy.data.images.load(depth_map_path)

normalize_node = tree.nodes.new(type="CompositorNodeNormalize")

defocus_node = tree.nodes.new(type="CompositorNodeDefocus")
defocus_node.use_zbuffer = True
defocus_node.f_stop = 32.0        
defocus_node.blur_max = 128       
defocus_node.bokeh = 'OCTAGON'

composite_node = tree.nodes.new(type="CompositorNodeComposite")

links = tree.links
links.new(image_node.outputs['Image'], defocus_node.inputs['Image'])
links.new(depth_node.outputs['Image'], normalize_node.inputs['Value'])
links.new(normalize_node.outputs['Value'], defocus_node.inputs['Z'])
links.new(defocus_node.outputs['Image'], composite_node.inputs['Image'])

scene.render.image_settings.file_format = 'JPEG'
scene.render.filepath = output_path

bpy.ops.render.render(write_still=True)

print(f"DoF blurred image saved at: {output_path}")
