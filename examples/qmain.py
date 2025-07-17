import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image  # You can also use PIL.Image.open
from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel

# Load local images
image = load_image("inpainting/templatemask.png")  # Replace with actual path
mask = load_image("inpainting/templatemask.png")      # Replace with actual path

# Load model
transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-fill-dev")
pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")

# Run inpainting
result = pipe(
    prompt="A wooden basket of a cat.",
    image=image,
    mask_image=mask,
    height=1024,
    width=1024,
    guidance_scale=30,
    num_inference_steps=50,
    max_sequence_length=512,
).images[0]

# Save output
result.save("flux.1-fill-dev.png")
