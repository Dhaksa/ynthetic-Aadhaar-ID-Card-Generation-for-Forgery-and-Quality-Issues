import bpy

bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree

for node in tree.nodes:
    tree.nodes.remove(node)

image_node = tree.nodes.new('CompositorNodeImage')
blur_node = tree.nodes.new('CompositorNodeBlur')
composite_node = tree.nodes.new('CompositorNodeComposite')
viewer_node = tree.nodes.new('CompositorNodeViewer')

image_path = r"J:\HyperGen\original.jpg"
image = bpy.data.images.load(image_path)
image_node.image = image

blur_node.filter_type = 'GAUSS'
blur_node.size_x = 10
blur_node.size_y = 10

links = tree.links
links.new(image_node.outputs['Image'], blur_node.inputs['Image'])
links.new(blur_node.outputs['Image'], composite_node.inputs['Image'])
links.new(blur_node.outputs['Image'], viewer_node.inputs['Image'])

bpy.context.scene.render.resolution_x = image.size[0]
bpy.context.scene.render.resolution_y = image.size[1]

output_path = r"J:\HyperGen\blurred.png"
bpy.context.scene.render.filepath = output_path
bpy.ops.render.render(use_viewport=False, write_still=True)
