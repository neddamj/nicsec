"""
Parameter sweep for patch_swap coefficients.
Standalone: duplicates required logic from patch_swap.py (no imports from it).
Saves only the best triplet as a single subplot image.
"""
import os
import csv
import time
import itertools
from typing import Dict, Any, List, Tuple, Optional

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
    "image_size": 256,
    "model_id": "my_bmshj2018_hyperprior",
    "quality": 1,
    "lr": 1e-3,
    "num_steps": 10000,
    "boxA": [100, 100, 50, 50],             # default: SVHN first digit bbox
    "boxB": [100, 100, 50, 50],             # default: boxA shifted right by its width
    "output_dir": "../outputs/patch_swap_sweep",
    "device": None,                         # "cuda" | "cpu" | None (auto)
    "sweep": {
        "patch_coef": [0.25, 0.5, 1.0],
        "bg_coef": [0.0, 0.05, 0.1],
        "tv_coef": [1e-5, 1e-4, 1e-3],
        "sim_coef": [0.1, 0.2, 0.4],
        "coarse_steps": 2000,
        "full_steps": 10000,
        "top_n": 5,
    },
}

# ---------------------------------------------------------------------
# Helpers (copied/adapted from patch_swap.py)
# ---------------------------------------------------------------------

def det_transform(image_size):
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((image_size, image_size)),
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
    return (
        torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])) +
        torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    )

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

def save_triplet_subplot(x_base, x_tgt, x_adv, bytes_base, bytes_tgt, bytes_adv, out_path):
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
    for ax_img, img, label in zip(axes, np_imgs, labels):
        ax_img.imshow(img)
        ax_img.axis("off")
        ax_img.set_title(label, fontsize=9, pad=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------
# Core sweep logic
# ---------------------------------------------------------------------

def choose_device(preferred: Optional[str] = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def prepare_fixed(cfg: Dict[str, Any]) -> Dict[str, Any]:
    image_size = cfg["image_size"]
    root_dir = os.path.join(cfg["data_root"], "train")

    img_base = Image.open(os.path.join(root_dir, cfg["base_name"])).convert("RGB")
    img_tgt = Image.open(os.path.join(root_dir, cfg["tgt_name"])).convert("RGB")

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

    boxes_stack = torch.stack([boxA_orig, boxB_orig], dim=0)
    x_base, boxes_resized = apply_det_transform(img_base, boxes_stack, image_size)
    x_tgt, _ = apply_det_transform(img_tgt, boxes_stack, image_size)
    boxA_rs, boxB_rs = boxes_resized[0], boxes_resized[1]

    xA1, yA1, xA2, yA2 = clamp_box_xyxy(boxA_rs, image_size)
    xB1, yB1, xB2, yB2 = clamp_box_xyxy(boxB_rs, image_size)

    if (xA2 - xA1 != xB2 - xB1) or (yA2 - yA1 != yB2 - yB1):
        raise ValueError("boxA and boxB must have same width and height after resize.")

    device = choose_device(cfg.get("device"))
    compressor = NeuralCompressor(model_id=cfg["model_id"], quality_factor=cfg["quality"], device=device)

    with torch.no_grad():
        x_tgt_d = x_tgt.unsqueeze(0).to(device)
        bytes_tgt = compressor.compress(x_tgt_d)["strings"][0][0]
        emb_tgt = compressor.compress_till_rounding(x_tgt_d)["y_hat"]
        emb_tgt_round = [torch.round(e) for e in emb_tgt]
        num_emb = sum(e.numel() for e in emb_tgt_round)
        bytes_base = compressor.compress(x_base.unsqueeze(0).to(device))["strings"][0][0]

    x_base_d = x_base.to(device)
    bg_mask = torch.ones((1, *x_base_d.shape), device=device)
    bg_mask[:, :, yA1:yA2, xA1:xA2] = 0
    bg_mask[:, :, yB1:yB2, xB1:xB2] = 0

    return {
        "device": device,
        "root_dir": root_dir,
        "x_base": x_base,
        "x_tgt": x_tgt,
        "x_base_d": x_base_d,
        "x_tgt_d": x_tgt_d,
        "compressor": compressor,
        "bytes_base": bytes_base,
        "bytes_tgt": bytes_tgt,
        "emb_tgt_round": emb_tgt_round,
        "num_emb": num_emb,
        "xA1": xA1,
        "yA1": yA1,
        "xA2": xA2,
        "yA2": yA2,
        "xB1": xB1,
        "yB1": yB1,
        "xB2": xB2,
        "yB2": yB2,
        "bg_mask": bg_mask,
    }


def compute_hamming(bytes_a: bytes, bytes_b: bytes) -> int:
    if len(bytes_a) == len(bytes_b):
        return sum(bin(b1 ^ b2).count("1") for b1, b2 in zip(bytes_a, bytes_b))
    return len(bytes_a) * 8


def run_once(
    run_cfg: Dict[str, Any],
    fixed: Dict[str, Any],
    num_steps: int,
    return_adv: bool = False,
) -> Dict[str, Any]:
    device = fixed["device"]
    compressor = fixed["compressor"]
    x_base = fixed["x_base_d"]

    x_adv = x_base.unsqueeze(0).to(device).clone().requires_grad_(True)
    opt = torch.optim.Adam([x_adv], lr=run_cfg["lr"])

    start = time.perf_counter()
    steps = 0
    last_hamm = None

    for it in range(num_steps):
        steps = it + 1
        temp = compressor.compress_till_rounding(x_adv)
        emb_adv = temp["y_hat"]

        loss_tgt = torch.zeros((1,), device=device)
        for ea, et in zip(emb_adv, fixed["emb_tgt_round"]):
            loss_tgt += torch.sum((ea - et) ** 2, dim=tuple(range(1, ea.dim())))
        loss_tgt = loss_tgt / fixed["num_emb"]

        loss_sim = F.mse_loss(x_adv, x_base.unsqueeze(0))

        patchA_adv = x_adv[0, :, fixed["yA1"]:fixed["yA2"], fixed["xA1"]:fixed["xA2"]]
        patchB_adv = x_adv[0, :, fixed["yB1"]:fixed["yB2"], fixed["xB1"]:fixed["xB2"]]
        patchA_base = x_base[:, fixed["yA1"]:fixed["yA2"], fixed["xA1"]:fixed["xA2"]]
        patchB_base = x_base[:, fixed["yB1"]:fixed["yB2"], fixed["xB1"]:fixed["xB2"]]

        loss_swapA = torch.mean((patchA_adv - patchB_base) ** 2)
        loss_swapB = torch.mean((patchB_adv - patchA_base) ** 2)

        loss_bg = torch.sum(fixed["bg_mask"] * (x_adv - x_base.unsqueeze(0))) / fixed["bg_mask"].sum().clamp(min=1)
        loss_tv = total_variation(x_adv)

        loss = (
            loss_tgt
            + run_cfg["patch_coef"] * (loss_swapA + loss_swapB)
            + run_cfg["bg_coef"] * loss_bg
            + run_cfg["tv_coef"] * loss_tv
            + run_cfg["sim_coef"] * loss_sim
        )

        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            x_adv.clamp_(0, 1)
            bytes_adv = compressor.compress(x_adv)["strings"][0][0]
            last_hamm = compute_hamming(bytes_adv, fixed["bytes_tgt"])

        if last_hamm == 0:
            break

    with torch.no_grad():
        bytes_adv_final = compressor.compress(x_adv)["strings"][0][0]
        hamming = compute_hamming(bytes_adv_final, fixed["bytes_tgt"])

        temp = compressor.compress_till_rounding(x_adv)
        emb_adv = temp["y_hat"]
        min_diff = torch.zeros((1,), device=device)
        for ea, et in zip(emb_adv, fixed["emb_tgt_round"]):
            min_diff = torch.maximum(min_diff, torch.amax(torch.abs(ea - et), dim=tuple(range(1, ea.dim()))))

        ssim = ssim_torch(x_adv, x_base.unsqueeze(0)).item()
        psnr = psnr_torch(x_adv, x_base.unsqueeze(0)).item()

    runtime_s = time.perf_counter() - start

    result = {
        "hamming": hamming,
        "minD": float(min_diff.item()),
        "ssim": float(ssim),
        "psnr": float(psnr),
        "steps": steps,
        "runtime_s": float(runtime_s),
    }
    if return_adv:
        result["x_adv"] = x_adv.detach().cpu().squeeze(0)
        result["bytes_adv"] = bytes_adv_final
    return result


def rank_results(results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    hamm0 = [r for r in results if r["hamming"] == 0]
    if hamm0:
        ranked = sorted(hamm0, key=lambda r: (-r["ssim"], -r["psnr"]))
        return ranked, "hamm0"
    ranked = sorted(results, key=lambda r: (r["hamming"], -r["ssim"], -r["psnr"]))
    return ranked, "fallback"


def run_sweep(cfg: Dict[str, Any]) -> None:
    fixed = prepare_fixed(cfg)
    sweep_cfg = cfg["sweep"]

    combos = list(itertools.product(
        sweep_cfg["patch_coef"],
        sweep_cfg["bg_coef"],
        sweep_cfg["tv_coef"],
        sweep_cfg["sim_coef"],
    ))

    results: List[Dict[str, Any]] = []

    print(f"Device: {fixed['device']}")
    print(f"Coarse sweep: {len(combos)} runs")

    for (pc, bg, tv, sc) in tqdm(combos, desc="Coarse sweep", dynamic_ncols=True):
        run_cfg = {
            "lr": cfg["lr"],
            "patch_coef": pc,
            "bg_coef": bg,
            "tv_coef": tv,
            "sim_coef": sc,
        }
        metrics = run_once(run_cfg, fixed, sweep_cfg["coarse_steps"], return_adv=False)
        results.append({
            "stage": "coarse",
            "patch_coef": pc,
            "bg_coef": bg,
            "tv_coef": tv,
            "sim_coef": sc,
            "num_steps": sweep_cfg["coarse_steps"],
            **metrics,
        })

    ranked, rank_mode = rank_results(results)
    if rank_mode == "fallback":
        print("WARNING: No Hamming==0 runs in coarse sweep; using fallback ranking.")

    top_n = min(sweep_cfg["top_n"], len(ranked))
    top_runs = ranked[:top_n]

    print(f"Full sweep: rerunning top {top_n} configs with {sweep_cfg['full_steps']} steps")
    full_results: List[Dict[str, Any]] = []
    for r in tqdm(top_runs, desc="Full sweep", dynamic_ncols=True):
        run_cfg = {
            "lr": cfg["lr"],
            "patch_coef": r["patch_coef"],
            "bg_coef": r["bg_coef"],
            "tv_coef": r["tv_coef"],
            "sim_coef": r["sim_coef"],
        }
        metrics = run_once(run_cfg, fixed, sweep_cfg["full_steps"], return_adv=True)
        full_results.append({
            "stage": "full",
            "patch_coef": r["patch_coef"],
            "bg_coef": r["bg_coef"],
            "tv_coef": r["tv_coef"],
            "sim_coef": r["sim_coef"],
            "num_steps": sweep_cfg["full_steps"],
            **metrics,
        })

    all_results = results + full_results

    if full_results:
        ranked_full, rank_mode_full = rank_results(full_results)
        best = ranked_full[0]
    else:
        ranked_full, rank_mode_full = rank_results(results)
        best_base = ranked_full[0]
        run_cfg = {
            "lr": cfg["lr"],
            "patch_coef": best_base["patch_coef"],
            "bg_coef": best_base["bg_coef"],
            "tv_coef": best_base["tv_coef"],
            "sim_coef": best_base["sim_coef"],
        }
        best = {
            "stage": "full",
            "patch_coef": best_base["patch_coef"],
            "bg_coef": best_base["bg_coef"],
            "tv_coef": best_base["tv_coef"],
            "sim_coef": best_base["sim_coef"],
            "num_steps": sweep_cfg["full_steps"],
        }
        best.update(run_once(run_cfg, fixed, sweep_cfg["full_steps"], return_adv=True))
        all_results.append(best)

    if rank_mode_full == "fallback":
        print("WARNING: No Hamming==0 runs in final selection; using fallback ranking.")

    output_dir = cfg["output_dir"]
    if not os.path.isabs(output_dir):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.normpath(os.path.join(base_dir, output_dir))
    os.makedirs(output_dir, exist_ok=True)

    best_path = os.path.join(output_dir, "best_triplet.png")
    save_triplet_subplot(
        fixed["x_base"].cpu(),
        fixed["x_tgt"].cpu(),
        best["x_adv"].cpu(),
        fixed["bytes_base"],
        fixed["bytes_tgt"],
        best["bytes_adv"],
        best_path,
    )

    csv_path = os.path.join(output_dir, "sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "stage",
            "patch_coef",
            "bg_coef",
            "tv_coef",
            "sim_coef",
            "num_steps",
            "hamming",
            "minD",
            "ssim",
            "psnr",
            "runtime_s",
        ])
        for r in all_results:
            writer.writerow([
                r["stage"],
                r["patch_coef"],
                r["bg_coef"],
                r["tv_coef"],
                r["sim_coef"],
                r["num_steps"],
                r["hamming"],
                f"{r['minD']:.6f}",
                f"{r['ssim']:.6f}",
                f"{r['psnr']:.6f}",
                f"{r['runtime_s']:.3f}",
            ])

    print(f"Saved best triplet to {best_path}")
    print(f"Saved sweep results to {csv_path}")
    print(
        "Best config: "
        f"patch={best['patch_coef']} bg={best['bg_coef']} tv={best['tv_coef']} sim={best['sim_coef']} | "
        f"H={best['hamming']} SSIM={best['ssim']:.4f} PSNR={best['psnr']:.2f}"
    )


def main():
    run_sweep(CONFIG)


if __name__ == "__main__":
    main()
