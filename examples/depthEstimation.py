import torch
import cv2
import numpy as np

input_image_path = "check.png"         
output_depth_path = "depth_mapcheck.png"    
model_type = "DPT_Large"

midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type in ["DPT_Large", "DPT_Hybrid"]:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

img = cv2.imread(input_image_path)
if img is None:
    raise FileNotFoundError(f"Image not found at path: {input_image_path}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img_rgb).to(device)

with torch.no_grad():
    prediction = midas(input_batch)

prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1),size=img.shape[:2],mode="bicubic",align_corners=False).squeeze()

depth = prediction.cpu().numpy()

depth_min = depth.min()
depth_max = depth.max()
print(f"Depth range: min={depth_min:.4f}, max={depth_max:.4f}")

depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_uint8 = depth_normalized.astype(np.uint8)

cv2.imwrite(output_depth_path, depth_uint8)
print(f"Depth map saved to: {output_depth_path}")
