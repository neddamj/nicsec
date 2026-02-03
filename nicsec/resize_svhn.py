"""
Resize numeric SVHN images, run a no-reference IQA gate (pyiqa), and save the
accepted 256x256 outputs. Edit CONFIG below to change paths, thresholds, or
logging. Requires: pip install pyiqa torch torchvision
"""
from pathlib import Path
from typing import List, Tuple, Optional
import re
import numpy as np
from PIL import Image

# -----------------
# Configuration
# -----------------
CONFIG = {
    # Paths
    "input_dir": "../data/svhn_full/train_extracted/train",
    "output_dir": "../data/svhn_256/train",

    # Selection + resize
    "count": 100,                   # 0 means all
    "size": 256,
    "pattern": r"\d+\.png",       # numeric filenames
    "overwrite": True,

    # Quality gate
    "quality_enabled": True,
    "metric_name": "musiq",       # e.g., musiq, nima, topiq_nr, paq2piq
    "gate_on": "original",        # "original" or "resized"
    "min_score": 25.0,              # used when higher is better
    "max_score": 60.0,              # used when lower is better

    # Logging / reporting
    "write_report_csv": False,
    "report_csv_path": "../data/svhn_256/quality_report.csv",
    "verbose": True,
}

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9
    RESAMPLE = Image.LANCZOS

# -----------------
# Helpers
# -----------------
def list_images(folder: Path, pattern: str) -> List[Path]:
    regex = re.compile(pattern)
    files = [p for p in folder.iterdir() if p.is_file() and regex.fullmatch(p.name)]
    return sorted(files, key=lambda p: int(p.stem))

def pil_to_tensor(img_rgb: Image.Image):
    import torch
    arr = np.asarray(img_rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

class NeuralIQA:
    """Thin wrapper around pyiqa metric."""
    def __init__(self, metric_name: str, device: Optional[str] = None):
        import torch
        import pyiqa

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.metric = pyiqa.create_metric(metric_name, device=self.device)
        self.lower_better = bool(getattr(self.metric, "lower_better", False))
        self.metric_name = metric_name

    def score(self, img_rgb: Image.Image) -> float:
        x = pil_to_tensor(img_rgb).to(self.device)
        with self.torch.no_grad():
            s = self.metric(x)
        return float(s.item()) if hasattr(s, "item") else float(s)

def quality_pass(score: float, lower_better: bool, cfg) -> Tuple[bool, str]:
    if lower_better:
        if cfg["max_score"] is None:
            return True, "ok (no max_score set)"
        if score <= cfg["max_score"]:
            return True, "ok"
        return False, f"score {score:.4f} > max_score {cfg['max_score']}"
    else:
        if cfg["min_score"] is None:
            return True, "ok (no min_score set)"
        if score >= cfg["min_score"]:
            return True, "ok"
        return False, f"score {score:.4f} < min_score {cfg['min_score']}"

def choose_gate_image(img: Image.Image, cfg) -> Image.Image:
    if cfg["gate_on"] == "resized":
        return img.resize((cfg["size"], cfg["size"]), RESAMPLE)
    return img

def process_images(files: List[Path], cfg) -> Tuple[int, int]:
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    report_rows = []
    if cfg.get("write_report_csv", False):
        report_rows.append(["filename", "kept", "metric", "score", "lower_better", "gate_on", "reason"])

    iqa = None
    if cfg["quality_enabled"]:
        iqa = NeuralIQA(cfg["metric_name"])
        direction = "LOWER is better" if iqa.lower_better else "HIGHER is better"
        print(f"[IQA] {cfg['metric_name']} on {iqa.device} ({direction})")

    saved = skipped = 0
    for src in files:
        dst = out_dir / src.name

        if dst.exists() and not cfg["overwrite"]:
            if cfg["verbose"]:
                print(f"skip {dst.name} (exists; set overwrite=True to replace)")
            skipped += 1
            if cfg.get("write_report_csv", False):
                report_rows.append([src.name, 0, cfg["metric_name"], "", "", cfg["gate_on"], "exists"])
            continue

        with Image.open(src) as img:
            img = img.convert("RGB")

            score = None
            reason = "quality_disabled"
            if cfg["quality_enabled"]:
                img_gate = choose_gate_image(img, cfg)
                score = iqa.score(img_gate)
                ok, reason = quality_pass(score, iqa.lower_better, cfg)
                if not ok:
                    print(f"reject {src.name} ({cfg['metric_name']}={score:.4f}; {reason})")
                    skipped += 1
                    if cfg.get("write_report_csv", False):
                        report_rows.append([src.name, 0, cfg["metric_name"], f"{score:.6f}", int(iqa.lower_better), cfg["gate_on"], reason])
                    continue
                if cfg["verbose"]:
                    print(f"keep   {src.name} ({cfg['metric_name']}={score:.4f}; {reason})")

            out = img.resize((cfg["size"], cfg["size"]), RESAMPLE)
            out.save(dst)
            saved += 1

            if cfg.get("write_report_csv", False):
                report_rows.append([
                    src.name,
                    1,
                    cfg["metric_name"],
                    "" if score is None else f"{score:.6f}",
                    "" if score is None else int(iqa.lower_better),
                    cfg["gate_on"],
                    reason,
                ])

    if cfg.get("write_report_csv", False):
        import csv
        report_path = Path(cfg["report_csv_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", newline="") as f:
            csv.writer(f).writerows(report_rows)
        print(f"[IQA] Wrote report: {report_path.resolve()}")

    return saved, skipped

# -----------------
# Entry point
# -----------------
def main():
    cfg = CONFIG
    in_dir = Path(cfg["input_dir"])

    if not in_dir.exists():
        raise SystemExit(f"Input directory not found: {in_dir}")

    files = list_images(in_dir, cfg["pattern"])
    if not files:
        raise SystemExit("No images matched the selection pattern.")

    selected = files if cfg["count"] <= 0 else files[: cfg["count"]]
    print(f"Found {len(files)} matches; processing {len(selected)} -> {cfg['output_dir']}")

    saved, skipped = process_images(selected, cfg)
    print(f"Done. Saved {saved}, skipped {skipped}. Output: {Path(cfg['output_dir']).resolve()}")

if __name__ == "__main__":
    main()