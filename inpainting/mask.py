import gradio as gr
import numpy as np
import imageio

def process_image(img):
    imageio.imwrite("output_image.png", img["composite"])

    alpha_channel = img["layers"][0][:, :, 3]
    mask = np.where(alpha_channel == 0, 0, 255)
    return mask

demo = gr.Interface(
    fn=process_image,
    inputs=gr.ImageMask(sources=["upload"], layers=False, transforms=[], format="png", label="base image", show_label=True),
    outputs=[gr.Image(label="Mask Image", format="png"),],
    description="Upload an image and then draw mask on the image")

demo.launch(debug=True)