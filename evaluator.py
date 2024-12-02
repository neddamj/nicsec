import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim
from typing import List, Dict
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb = None
    wandb_available = False

class MetricsEvaluator:
    def __init__(self, batch_size: int, use_wandb: bool = False):
        self.batch_size = batch_size
        self.use_wandb = use_wandb

        # Metrics storage
        self.success_rates = []
        self.l2_distances = []
        self.ssim_values = []
        self.psnr_values = []

    def calculate_metrics(self, img1: torch.Tensor, img2: torch.Tensor) -> Dict:
        """
        Calculate PSNR, SSIM, and L2 distance between img1 and img2 (batch-wise).
        """
        normalized_l2_distances = []
        structural_similarities = []
        psnr_values = []
        
        for i in range(self.batch_size):
            normalized_l2_distances.append(self._l2_norm(img1, img2[i]))
            structural_similarities.append(self._ssim(img1, img2[i]))
            psnr_values.append(self._psnr(img1, img2[i]))
        
        return {
            'l2': torch.tensor(normalized_l2_distances).mean(),
            'ssim': torch.tensor(structural_similarities).mean(),
            'psnr': torch.tensor(psnr_values).mean()
        }

    def _psnr(self, img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
        mse = F.mse_loss(img1, img2)
        psnr = 10 * torch.log10(max_val / torch.sqrt(mse))
        return psnr

    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return ms_ssim(img1, img2.unsqueeze(0), data_range=1.0)

    def _l2_norm(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Calculate the normalized L2 distance between two images.
        """
        return torch.norm(img1 - img2) / img1.numel()

    def evaluate_batch(self, src_img: torch.Tensor, adv_imgs: torch.Tensor, output: torch.Tensor, model: torch.nn.Module) -> None:
        """
        Evaluate a batch and store the success rate and calculated metrics.
        """
        # Calculate success rate
        success_rate = self.calculate_success_rate(src_img, adv_imgs, model)
        self.success_rates.append(success_rate)

        # Calculate PSNR, SSIM, and L2
        metrics = self.calculate_metrics(src_img, output)
        self.l2_distances.append(metrics['l2'])
        self.ssim_values.append(metrics['ssim'])
        self.psnr_values.append(metrics['psnr'])

        # Log batch-wise metrics to wandb
        if self.use_wandb:
            wandb.log({
                'batch_success_rate': success_rate / self.batch_size,
                'batch_l2': metrics['l2'].item(),
                'batch_ssim': metrics['ssim'].item(),
                'batch_psnr': metrics['psnr'].item()
            })
        else:
            print(f"BATCH METRICS\nSuccess: {success_rate}/{self.batch_size}\nNormalized L2 Dist: {metrics['l2']}\nSSIM: {metrics['ssim']}\nPSNR: {metrics['psnr']}\n")

    def global_metrics(self) -> None:
        """
        Calculate and print the global average of success rate, PSNR, SSIM, and L2 distance.
        """
        asr = self._calculate_global_average(self.success_rates) / self.batch_size
        l2 = self._calculate_global_average(self.l2_distances)
        ssim = self._calculate_global_average(self.ssim_values)
        psnr = self._calculate_global_average(self.psnr_values)

        # Log global metrics to wandb
        if self.use_wandb:
            wandb.log({
                'global_asr': asr,
                'global_l2': l2,
                'global_ssim': ssim,
                'global_psnr': psnr
            })
        else:
            print(f"\nGLOBAL METRICS\nASR: {asr}\nNormalized L2 Dist: {l2}\nSSIM: {ssim}\nPSNR: {psnr}")

    def calculate_success_rate(self, src_img: torch.Tensor, adv_imgs: torch.Tensor, model: torch.nn.Module) -> int:
        """
        Calculate the success rate of the attack by measuring the hamming distance between the compressed
        versions of src_img and adv_imgs.
        """
        success = 0
        with torch.no_grad():
            src_bytes = model.compress(src_img)['strings'][0][0]
            adv_bytes_list = model.compress(adv_imgs)['strings'][0]
        for adv_bytes in adv_bytes_list:
            hamming_dist = 0
            if len(adv_bytes) != len(src_bytes):
                continue
            for x_byte, adv_byte in zip(src_bytes, adv_bytes):
                hamming_dist += bin(x_byte ^ adv_byte).count('1')
            if hamming_dist == 0:
                success += 1
        return success

    def _calculate_global_average(self, metric: List[float]) -> float:
        return sum(metric) / len(metric) if metric else 0.0