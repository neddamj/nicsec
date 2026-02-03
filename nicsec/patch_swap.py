"""
Two-patch swap collision attack.
Swaps the content of boxA <-> boxB while matching the target bitstream.
"""
import os
from datetime import datetime

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as T
from torchvision.tv_tensors import BoundingBoxes, Image as TVImage
from tqdm import tqdm

from compressor import NeuralCompressor
from dataset import SVHNFullBBox

# ---------------------------------------------------------------------
# Config (edit manually)
# ---------------------------------------------------------------------
img_id = 88
CONFIG = {
    "data_root": "/home/jmadden2/Documents/Research/nicsec/data/svhn_256",
    "base_name": f"{img_id}_edit.png",      # appearance starts from this
    "tgt_name":  f"{img_id}.png",           # supplies target bitstream
    "save_name": f"{img_id}_swapped.png",
    "image_size": 256,
    "model_id": "my_bmshj2018_hyperprior",
    "quality": 1,
    "lr": 1e-3,
    "num_steps": 10000,
    "patch_coef": 0.5,                      # weight for swap losses
    "bg_coef": 0,                           # weight for keeping background near base image
    "tv_coef": 1e-4,                        # helps hide seams
    "sim_coef": 0.2,                        # weight for overall similarity to base image
    "viz_name": f"{img_id}_viz.png",        # saved grid of base/target/adv with bitstreams; set to None to skip
    "output_dir": "../outputs/patch_swap",
    # Set boxes in ORIGINAL pixel coords (x1,y1,x2,y2). If None, auto choices:
    "boxA": [100, 100, 50, 50],             # default: SVHN first digit bbox
    "boxB": [100, 100, 50, 50],             # default: boxA shifted right by its width
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

def det_transform(image_size):
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((image_size, image_size))
    ])

def apply_det_transform(img, boxes_xyxy, image_size):
    boxes = BoundingBoxes(boxes_xyxy, format="XYXY", canvas_size=(img.height, img.width))
    img_tv = TVImage(img)
    img_tv, boxes = det_transform(image_size)(img_tv, boxes)
    return img_tv.data, torch.as_tensor(boxes, dtype=torch.float32)

def load_svhn_bbox(data_root, name):
    ds = SVHNFullBBox(root=data_root, split="train", transform=None)
    for rec in ds.records:
        if rec["name"] == name:
            return rec["boxes"][0]  # first digit bbox
    raise ValueError(f"bbox for {name} not found")

def clamp_box_xyxy(box, size):
    # Accept tensor or list; convert to python floats before rounding
    if torch.is_tensor(box):
        box_vals = [float(b) for b in box]
    else:
        box_vals = list(box)
    x1 = max(0, min(size - 1, int(round(box_vals[0]))))
    y1 = max(0, min(size - 1, int(round(box_vals[1]))))
    x2 = max(x1 + 1, min(size,     int(round(box_vals[2]))))
    y2 = max(y1 + 1, min(size,     int(round(box_vals[3]))))
    return x1, y1, x2, y2

def total_variation(x):
    return (torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])) +
            torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])))


# ---------------------------------------------------------------------
# Quality metrics (adv should look like base)
# ---------------------------------------------------------------------
def _gaussian_kernel(channels, kernel_size=11, sigma=1.5, device=None, dtype=None):
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
    return kernel


@torch.no_grad()
def ssim_torch(x, y, kernel_size=11, sigma=1.5, c1=0.01 ** 2, c2=0.03 ** 2):
    """
    Structural Similarity Index for tensors in [0,1].
    Inputs: x, y shaped (N, C, H, W) on same device.
    Returns scalar tensor.
    """
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError("ssim_torch expects tensors shaped (N,C,H,W)")
    channels = x.size(1)
    kernel = _gaussian_kernel(channels, kernel_size, sigma, device=x.device, dtype=x.dtype)
    padding = kernel_size // 2
    mu_x = F.conv2d(x, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=channels) - mu_xy

    ssim_n = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = ssim_n / ssim_d
    return ssim_map.mean()


@torch.no_grad()
def psnr_torch(x, y, max_val=1.0):
    mse = F.mse_loss(x, y)
    if mse == 0:
        return torch.tensor(float("inf"), device=x.device)
    return 10 * torch.log10(max_val ** 2 / mse)

def first_hex_digits(bts, num=30):
    hx = bts.hex()
    return hx[:num] if hx else "(empty bitstream)"

def save_trips_with_bitstreams(x_base, x_tgt, x_adv, bytes_base, bytes_tgt, bytes_adv, out_path):
    """
    Save a single image containing base / target / adversarial images with their bitstreams above.
    Inputs are tensors shaped (C,H,W) on CPU.
    """
    imgs = [x_base, x_tgt, x_adv]
    labels = [
        f"base: {first_hex_digits(bytes_base, num=30)}...",
        f"target: {first_hex_digits(bytes_tgt, num=30)}...",
        f"adv: {first_hex_digits(bytes_adv, num=30)}...",
    ]

    np_imgs = [img.detach().clamp(0, 1).permute(1, 2, 0).numpy() for img in imgs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 10))

    # Bottom row: images
    for ax_img, img, label in zip(axes, np_imgs, labels):
        ax_img.imshow(img)
        ax_img.axis("off")
        ax_img.set_title(label, fontsize=9, pad=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
cfg = CONFIG
image_size = cfg["image_size"]
root_dir = os.path.join(cfg["data_root"], "train")

# Load images
img_base = Image.open(os.path.join(root_dir, cfg["base_name"])).convert("RGB")
img_tgt  = Image.open(os.path.join(root_dir, cfg["tgt_name"])).convert("RGB")

# Boxes in original coords
if cfg["boxA"] is None:
    boxA_orig = load_svhn_bbox(cfg["data_root"], cfg["base_name"])
else:
    boxA_orig = torch.tensor(cfg["boxA"], dtype=torch.float32)

if cfg["boxB"] is None:
    w = boxA_orig[2] - boxA_orig[0]
    boxB_orig = boxA_orig.clone()
    boxB_orig[0] += w
    boxB_orig[2] += w
else:
    boxB_orig = torch.tensor(cfg["boxB"], dtype=torch.float32)

# Resize images + boxes consistently
boxes_stack = torch.stack([boxA_orig, boxB_orig], dim=0)
x_base, boxes_resized = apply_det_transform(img_base, boxes_stack, image_size)
x_tgt,  _             = apply_det_transform(img_tgt,  boxes_stack, image_size)
boxA_rs, boxB_rs = boxes_resized[0], boxes_resized[1]

# Clamp to ints inside canvas
xA1, yA1, xA2, yA2 = clamp_box_xyxy(boxA_rs, image_size)
xB1, yB1, xB2, yB2 = clamp_box_xyxy(boxB_rs, image_size)

# Sanity: same size patches
if (xA2 - xA1 != xB2 - xB1) or (yA2 - yA1 != yB2 - yB1):
    raise ValueError("boxA and boxB must have same width and height after resize.")

# Compressor + targets
compressor = NeuralCompressor(model_id=cfg["model_id"], quality_factor=cfg["quality"], device=device)
with torch.no_grad():
    x_tgt = x_tgt.unsqueeze(0).to(device)
    bytes_tgt = compressor.compress(x_tgt)["strings"][0][0]
    emb_tgt = compressor.compress_till_rounding(x_tgt)["y_hat"]
    emb_tgt_round = [torch.round(e) for e in emb_tgt]
    num_emb = sum(e.numel() for e in emb_tgt_round)
    bytes_base = compressor.compress(x_base.unsqueeze(0).to(device))["strings"][0][0]

# Initialize adversarial image
x_adv = x_base.unsqueeze(0).to(device).clone().requires_grad_(True)
x_base = x_base.to(device)

opt = torch.optim.Adam([x_adv], lr=cfg["lr"])
start = datetime.now()

pbar = tqdm(range(cfg["num_steps"]), desc="Optimizing", dynamic_ncols=True)
for it in pbar:
    temp = compressor.compress_till_rounding(x_adv)
    emb_adv = temp["y_hat"]

    loss_tgt = torch.zeros((1,), device=device)
    for ea, et in zip(emb_adv, emb_tgt_round):
        loss_tgt += torch.sum((ea - et) ** 2, dim=tuple(range(1, ea.dim())))
    loss_tgt = loss_tgt / num_emb

    loss_sim = F.mse_loss(x_adv, x_base.unsqueeze(0))

    # Swap losses
    patchA_adv = x_adv[0, :, yA1:yA2, xA1:xA2]
    patchB_adv = x_adv[0, :, yB1:yB2, xB1:xB2]
    patchA_base = x_base[:, yA1:yA2, xA1:xA2]
    patchB_base = x_base[:, yB1:yB2, xB1:xB2]

    loss_swapA = torch.mean((patchA_adv - patchB_base) ** 2)  # A should look like original B
    loss_swapB = torch.mean((patchB_adv - patchA_base) ** 2)  # B should look like original A

    # Background fidelity (outside both patches)
    bg_mask = torch.ones_like(x_adv)
    bg_mask[:, :, yA1:yA2, xA1:xA2] = 0
    bg_mask[:, :, yB1:yB2, xB1:xB2] = 0
    loss_bg = torch.sum(bg_mask * (x_adv - x_base)) / bg_mask.sum().clamp(min=1)

    loss_tv = total_variation(x_adv)

    loss = (
        loss_tgt
        + cfg["patch_coef"] * (loss_swapA + loss_swapB)
        + cfg["bg_coef"] * loss_bg
        + cfg["tv_coef"] * loss_tv
        + cfg["sim_coef"] * loss_sim
    )

    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        x_adv.clamp_(0, 1)

    # Hamming distance check
    with torch.no_grad():
        bytes_adv = compressor.compress(x_adv)["strings"][0][0]
        if len(bytes_adv) == len(bytes_tgt):
            hamm = sum(bin(b1 ^ b2).count("1") for b1, b2 in zip(bytes_adv, bytes_tgt))
        else:
            hamm = len(bytes_adv) * 8
        min_diff = torch.zeros((1,), device=device)
        for ea, et in zip(emb_adv, emb_tgt_round):
            min_diff = torch.maximum(min_diff,
                                     torch.amax(torch.abs(ea - et), dim=tuple(range(1, ea.dim()))))
    pbar.set_postfix({
        "loss": f"{loss.item():.4f}",
        "tgt": f"{loss_tgt.item():.4f}",
        "swap": f"{(loss_swapA+loss_swapB).item():.4f}",
        "bg": f"{loss_bg.item():.4f}",
        "H": int(hamm),
        "minD": f"{min_diff.item():.4f}",
    })
    if hamm == 0:
        pbar.write("Hamming distance zero achieved; stopping.")
        break

# Final bitstream + visualization
with torch.no_grad():
    bytes_adv_final = compressor.compress(x_adv)["strings"][0][0]

# Quality metrics (adv vs base)
with torch.no_grad():
    ssim_adv = ssim_torch(x_adv, x_base.unsqueeze(0)).item()
    psnr_adv = psnr_torch(x_adv, x_base.unsqueeze(0)).item()
print(f"SSIM(base, adv): {ssim_adv:.4f}; PSNR(base, adv): {psnr_adv:.2f}dB")

if cfg.get("viz_name"):
    output_dir, viz_path = cfg["output_dir"], cfg["viz_name"]
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isabs(viz_path):
        viz_path = os.path.join(output_dir, viz_path)
    save_trips_with_bitstreams(
        x_base.detach().cpu(),
        x_tgt.detach().cpu().squeeze(0),
        x_adv.detach().cpu().squeeze(0),
        bytes_base,
        bytes_tgt,
        bytes_adv_final,
        viz_path,
    )
    print(f"Saved visualization to {viz_path}")

# Save
out_img = (x_adv.detach().cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(out_img).save(os.path.join(root_dir, cfg["save_name"]))
print(f"\nSaved optimized image to {os.path.join(root_dir, cfg['save_name'])}")
print(f"Total time {(datetime.now()-start).total_seconds():.1f}s")
