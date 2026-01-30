"""
Generate a colliding adversarial image from a target image by scrambling a patch.
Simplified: static CONFIG dict, minimal imports, tqdm progress, saved visualization.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import List

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2 as T
from torchvision.tv_tensors import BoundingBoxes, Image as TVImage
from tqdm.auto import tqdm

from compressor import JpegCompressor, NeuralCompressor
from dataset import KodakDataset, SVHNFullBBox, WiderFaceDataset, svhn_collate


# ---------------------------------------------------------------------------#
# Settings (edit these)
# ---------------------------------------------------------------------------#
CONFIG = {
    "lr": 1e-3,
    "batch_size": 8,
    "num_batches": 1,
    "num_steps": 5000,
    "model_id": "my_bmshj2018_hyperprior",  #"my_bmshj2018_factorized_relu", "my_bmshj2018_factorized", "my_mbt2018_mean", "my_cheng2020_anchor"
    "quality_factor": 1,
    "compressor_type": "neural",      # "neural" | "jpeg"
    "image_size": 256,
    "dataset": "wider-face",                # "imagenette", "svhn", "wider-face", "celeba", "kodak"
    "num_patches": 1,
    "patch_size": 10,
    "scramble_factor": 0.05,
    "visualize": True,
    "output_dir": "../outputs/patch_scramble",
    "device": None,                   # "cuda" | "mps" | "cpu" | None (auto)
}

# ---------------------------------------------------------------------------#
# Utilities
# ---------------------------------------------------------------------------#
def choose_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def set_cpu_threads():
    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)

def build_compressor(cfg, device: str):
    if cfg["compressor_type"] == "neural":
        return NeuralCompressor(cfg["model_id"], cfg["quality_factor"], device=device)
    return JpegCompressor(
        differentiable=True,
        quality_factor=cfg["quality_factor"],
        image_size=cfg["image_size"],
        device=device,
    )

def build_transforms(cfg):
    image_tf = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
    ])

    det_tf = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((cfg["image_size"], cfg["image_size"])),
    ])

    def apply_det_transform(img, target):
        boxes = BoundingBoxes(
            target["boxes"],
            format="XYXY",
            canvas_size=(img.height, img.width),
        )
        img_tv = TVImage(img)
        img_tv, boxes = det_tf(img_tv, boxes)
        target = dict(target)
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        return img_tv.data, target

    return image_tf, apply_det_transform

def build_dataloader(cfg, image_tf, det_tf):
    if cfg["dataset"] == "celeba":
        ds = datasets.CelebA(root="../data", split="train", transform=image_tf, download=True)
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

    if cfg["dataset"] == "imagenette":
        ds = datasets.Imagenette(root="../data", split="train", transform=image_tf)
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

    if cfg["dataset"] == "kodak":
        ds = KodakDataset(root="../data/kodak", transform=image_tf)
        return DataLoader(ds, batch_size=24, shuffle=True)  # keep original bump

    if cfg["dataset"] == "wider-face":
        pipeline = WiderFaceDataset(
            image_size=cfg["image_size"],
            batch_size=cfg["batch_size"],
            min_ratio=0.04,
            max_ratio=0.05,
            split="validation",
        )
        return pipeline.get_dataloader(shuffle=True)

    if cfg["dataset"] == "svhn":
        train_ds = SVHNFullBBox(
            root="../data/svhn_full/train_extracted",
            split="train",
            transform=det_tf,
        )
        return DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=svhn_collate)

    raise ValueError(f"Unknown dataset: {cfg['dataset']}")

# ---------------------------------------------------------------------------#
# Core helpers
# ---------------------------------------------------------------------------#
def prepare_targets(compressor, x_tgt: torch.Tensor):
    with torch.no_grad():
        bytes_tgt_list = compressor.compress(x_tgt)["strings"][0]
        temp = compressor.compress_till_rounding(x_tgt)
        emb_tgt = temp["y_hat"]
        emb_tgt_round = [torch.round(item) for item in emb_tgt]
        num_emb = sum(e[0].numel() for e in emb_tgt_round)
    return bytes_tgt_list, emb_tgt, emb_tgt_round, num_emb

def build_patches(cfg, bboxes, batch_size: int, device: str):
    """Return num_patches, patches[B*num_patches, 4] in XYWH."""
    if cfg["dataset"] == "wider-face":
        num_patches = 1
        patches = torch.zeros((batch_size * num_patches, 4), dtype=torch.int32, device=device)
        for i in range(batch_size):
            patches[i * num_patches:(i + 1) * num_patches] = bboxes[i][:num_patches]
        return num_patches, patches

    if cfg["dataset"] == "svhn":
        num_patches = 1
        patches = torch.zeros((batch_size * num_patches, 4), dtype=torch.int32, device=device)
        for i in range(batch_size):
            patches[i * num_patches:(i + 1) * num_patches] = bboxes[0]["boxes"][:num_patches]
        return num_patches, patches

    num_patches = cfg["num_patches"]
    patch_size = cfg["patch_size"]
    centers = np.random.randint(patch_size, high=cfg["image_size"] - patch_size, size=(num_patches, 2))
    patches_list = [
        [c[0] - patch_size, c[1] - patch_size, 2 * patch_size, 2 * patch_size]
        for c in centers
    ]
    patches = torch.tensor(patches_list, device=device).repeat(batch_size, 1)
    return num_patches, patches

def expand_targets(num_patches, x_tgt_base, emb_tgt, emb_tgt_round):
    x_tgt = x_tgt_base.repeat_interleave(num_patches, dim=0)
    emb_tgt = [e.repeat_interleave(num_patches, dim=0) for e in emb_tgt]
    emb_tgt_round = [e.repeat_interleave(num_patches, dim=0) for e in emb_tgt_round]
    return x_tgt, emb_tgt, emb_tgt_round

def loss_and_metrics(emb_adv, target_info, x_adv, patches, scramble_factor):
    device = x_adv.device
    B2 = patches.shape[0]
    L = len(emb_adv)

    loss_tgt = torch.zeros((B2,), device=device)
    for i in range(L):
        diff = emb_adv[i] - target_info["emb_tgt_round"][i]
        loss_tgt += torch.sum(diff * diff, dim=tuple(range(1, diff.dim())))
    loss_tgt = loss_tgt / target_info["num_emb"]

    loss_con = torch.zeros((B2,), device=device)
    for i in range(B2):
        xs = x_adv[i, :, patches[i, 1]:patches[i, 1] + patches[i, 3], patches[i, 0]:patches[i, 0] + patches[i, 2]]
        xt = target_info["x_tgt"][i, :, patches[i, 1]:patches[i, 1] + patches[i, 3], patches[i, 0]:patches[i, 0] + patches[i, 2]]
        loss_con[i] = torch.mean((xs - xt) ** 2)

    total_loss = torch.sum(loss_tgt - scramble_factor * loss_con)
    return total_loss, loss_tgt, loss_con

def max_channel_abs_diff(tensors_a: List[torch.Tensor], tensors_b: List[torch.Tensor]):
    device = tensors_a[0].device
    B2 = tensors_a[0].shape[0]
    min_diff = torch.zeros((B2,), device=device)
    for a, b in zip(tensors_a, tensors_b):
        min_diff = torch.maximum(min_diff, torch.amax(torch.abs(a - b), dim=tuple(range(1, a.dim()))))
    return min_diff

def compute_hamming_zero(compressor, x_adv, bytes_tgt_list, num_patches: int, B: int, device: str):
    bytes_adv_list = compressor.compress(x_adv)["strings"][0]
    hamm = []
    for i in range(B):
        for adv_bytes in bytes_adv_list[i * num_patches:(i + 1) * num_patches]:
            if len(bytes_tgt_list[i]) != len(adv_bytes):
                hamm.append(len(adv_bytes) * 8)
                continue
            hamming_dist = sum(bin(tb ^ ab).count("1") for tb, ab in zip(bytes_tgt_list[i], adv_bytes))
            hamm.append(hamming_dist)
    hamm = torch.tensor(hamm, device=device)
    return torch.where(hamm == 0)[0], bytes_adv_list

# ---------------------------------------------------------------------------#
# Main attack loop
# ---------------------------------------------------------------------------#
def run_attack(cfg, dataloader: DataLoader, compressor, device: str):
    best_result = None
    target_info = None
    steps_range = range(cfg["num_steps"])

    for batch_idx, (x_src, bboxes) in enumerate(dataloader):
        if cfg["dataset"] == "svhn":
            x_src = torch.stack(x_src)

        if target_info is None:
            x_tgt_base = x_src.clone().to(device)
            bytes_tgt, emb_tgt, emb_tgt_round, num_emb = prepare_targets(compressor, x_tgt_base)
            num_patches, patches = build_patches(cfg, bboxes, x_tgt_base.shape[0], device)
            x_tgt, emb_tgt, emb_tgt_round = expand_targets(num_patches, x_tgt_base, emb_tgt, emb_tgt_round)
            target_info = dict(
                x_tgt=x_tgt,
                emb_tgt=emb_tgt,
                emb_tgt_round=emb_tgt_round,
                bytes_tgt=bytes_tgt,
                num_emb=num_emb,
                num_patches=num_patches,
                patches=patches,
                base_B=x_tgt_base.shape[0],
            )

        B = target_info["base_B"]
        num_patches = target_info["num_patches"]
        patches = target_info["patches"]
        B2 = B * num_patches

        tqdm.write(
            f'batch {batch_idx}: model={cfg["model_id"]}, qf={cfg["quality_factor"]}, lr={cfg["lr"]}, '
            f'gpu={device}, num_patches={num_patches}, date={datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        )

        x_adv = x_src.clone().to(device).repeat_interleave(num_patches, dim=0)
        x_adv.requires_grad = True
        optimizer = torch.optim.Adam([x_adv], lr=cfg["lr"])

        progress = tqdm(steps_range, desc=f"batch{batch_idx}", leave=False)
        for step in progress:
            temp = compressor.compress_till_rounding(x_adv)
            emb_adv = temp["y_hat"]

            loss, loss_tgt, loss_con = loss_and_metrics(
                emb_adv, target_info, x_adv, patches, cfg["scramble_factor"]
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # clamp without tracking grads to avoid in-place op on leaf with grad
            with torch.no_grad():
                x_adv.clamp_(0, 1)

            with torch.no_grad():
                min_diff = max_channel_abs_diff(emb_adv, target_info["emb_tgt_round"])
                if min_diff.min() > 0.5:
                    hamm0_list = torch.tensor([], device=device)
                    bytes_adv_list = None
                    progress.set_postfix(loss=f"{loss.item():.4f}", h0=f"0/{B2}", mindiff=f"{min_diff.min().item():.3f}")
                    continue

                hamm0_list, bytes_adv_list = compute_hamming_zero(
                    compressor, x_adv, target_info["bytes_tgt"], num_patches, B, device
                )

                if len(hamm0_list) == 0:
                    progress.set_postfix(loss=f"{loss.item():.4f}", h0=f"0/{B2}", mindiff=f"{min_diff.min().item():.3f}")
                    continue

                idx = hamm0_list[torch.argmax(loss_con[hamm0_list])]
                patch_tgt = target_info["x_tgt"][idx,
                                                 :,
                                                 patches[idx, 1]:patches[idx, 1] + patches[idx, 3],
                                                 patches[idx, 0]:patches[idx, 0] + patches[idx, 2]]
                mse_in_theory = torch.mean(1 / 12 + (patch_tgt - 0.5) ** 2).item()
                mse_in_patch = torch.mean((patch_tgt - x_adv[idx,
                                                             :,
                                                             patches[idx, 1]:patches[idx, 1] + patches[idx, 3],
                                                             patches[idx, 0]:patches[idx, 0] + patches[idx, 2]]) ** 2).item()

                best_result = dict(
                    idx=idx.item(),
                    loss_con=loss_con[idx].item(),
                    mse_in_patch=mse_in_patch,
                    mse_in_theory=mse_in_theory,
                    x_adv=x_adv[idx].detach().clone(),
                    x_tgt=target_info["x_tgt"][idx].detach().clone(),
                    bbox=patches[idx].clone(),
                    bytes_adv=bytes_adv_list[idx],
                    bytes_tgt=target_info["bytes_tgt"][idx // num_patches],
                )

                progress.set_postfix(loss=f"{loss.item():.4f}", h0=f"{len(hamm0_list)}/{B2}", mse=f"{mse_in_patch:.3f}")

                if mse_in_patch >= mse_in_theory * 2:
                    break

        progress.close()
        if batch_idx >= cfg["num_batches"] - 1:
            break

    return best_result, num_patches

# ---------------------------------------------------------------------------#
# Visualization
# ---------------------------------------------------------------------------#
def visualize_best(best_result, num_patches: int, save_path: str | None = None, show: bool = True):
    if best_result is None:
        print("No collision found; nothing to visualize.")
        return

    x_adv_np = best_result["x_adv"].permute(1, 2, 0).cpu().numpy()
    x_tgt_np = best_result["x_tgt"].permute(1, 2, 0).cpu().numpy()
    bbox = best_result["bbox"].cpu().numpy()

    print(
        f'best colliding adv image is # {best_result["idx"]} '
        f'(image {best_result["idx"] // num_patches}, patch {best_result["idx"] % num_patches}), '
        f'Patch MSE={best_result["mse_in_patch"]:.4f}, '
        f'whole image MSE={np.mean(np.square(x_adv_np - x_tgt_np)):.4f}'
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(x_adv_np); ax1.axis("off")
    ax1.set_title("adv: " + best_result["bytes_adv"].hex()[:15] + "...")

    ax2.imshow(x_tgt_np)
    from matplotlib.patches import Rectangle
    rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                     linewidth=1, edgecolor="red", facecolor="none")
    ax2.add_patch(rect)
    ax2.axis("off")
    ax2.set_title("tgt: " + best_result["bytes_tgt"].hex()[:15] + "...")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"saved visualization to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    cfg = CONFIG
    device = choose_device(cfg["device"])
    if device == "cpu":
        set_cpu_threads()
    print("device:", device)

    compressor = build_compressor(cfg, device)
    print("compressor:", type(compressor).__name__)

    image_tf, det_tf = build_transforms(cfg)
    dataloader = build_dataloader(cfg, image_tf, det_tf)
    print("dataset:", cfg["dataset"])

    best_result, num_patches = run_attack(cfg, dataloader, compressor, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = None
    if cfg["output_dir"]:
        os.makedirs(cfg["output_dir"], exist_ok=True)
        save_path = os.path.join(cfg["output_dir"], f"best_{timestamp}.png")

    if cfg["visualize"] or save_path:
        visualize_best(best_result, num_patches, save_path=save_path, show=cfg["visualize"])


if __name__ == "__main__":
    main()
