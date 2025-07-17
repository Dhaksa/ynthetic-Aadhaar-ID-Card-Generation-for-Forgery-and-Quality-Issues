import torch
import gc
from PIL import Image
from diffusers import FluxFillPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

gc.collect()
torch.cuda.empty_cache()

GGUF_PATH = "flux1-fill-dev-Q3_K_S.gguf"
INPUT_IMAGE_PATH = "template.png"
MASK_IMAGE_PATH = "templatemask.png"
OUTPUT_PATH = "templateoutput.png"

transformer = FluxTransformer2DModel.from_single_file(GGUF_PATH,quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),torch_dtype=torch.float16,).to("cuda")

pipeline = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev",transformer=transformer,torch_dtype=torch.bfloat16,).to("cuda")

pipeline.enable_model_cpu_offload() 

base_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
mask_image = Image.open(MASK_IMAGE_PATH).convert("L")

prompt = (
    "Replace the existing name with 'VISHWA', preserving the exact same font, "
    "size, style, and alignment. Ensure the new text seamlessly matches the "
    "surrounding background and lighting so that it looks natural and unaltered."
)

result = pipeline(
    prompt=prompt,
    image=base_image,
    mask_image=mask_image,
    height=512,
    width=512,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cuda").manual_seed(1234),
).images[0]

result.save(OUTPUT_PATH)
print(f"Output saved at: {OUTPUT_PATH}")
