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
        self.input_l2_distances = []
        self.input_ssim_values = []
        self.input_psnr_values = []
        self.output_l2_distances = []
        self.output_ssim_values = []
        self.output_psnr_values = []

    def calculate_input_metrics(self, img1: torch.Tensor, img2: torch.Tensor) -> Dict:
        """
        Calculate PSNR, SSIM, and L2 distance between img1 and img2 (batch-wise).
        """
        normalized_l2_distances = []
        structural_similarities = []
        psnr_values = []
        for i in range(self.batch_size):
            normalized_l2_distances.append(self._l2_norm(img1[i].unsqueeze(0), img2[i]))
            structural_similarities.append(self._ssim(img1[i].unsqueeze(0), img2[i]))
            psnr_values.append(self._psnr(img1[i].unsqueeze(0), img2[i]))
        
        return {
            'l2': torch.tensor(normalized_l2_distances).mean(),
            'ssim': torch.tensor(structural_similarities).mean(),
            'psnr': torch.tensor(psnr_values).mean()
        }
    
    def calculate_output_metrics(self, img1: torch.Tensor, img2: torch.Tensor) -> Dict:
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

    def evaluate_batch(self, src_img: torch.Tensor, target_img: torch.Tensor, adv_imgs: torch.Tensor, output: torch.Tensor, model: torch.nn.Module) -> None:
        """
        Evaluate a batch and store the success rate and calculated metrics.
        """
        # Calculate success rate
        success_rate = self.calculate_success_rate(target_img, adv_imgs, model)
        self.success_rates.append(success_rate)

        # Calculate PSNR, SSIM, and L2 between the adv images and the target image
        output_metrics = self.calculate_output_metrics(target_img, adv_imgs)
        self.output_l2_distances.append(output_metrics['l2'])
        self.output_ssim_values.append(output_metrics['ssim'])
        self.output_psnr_values.append(output_metrics['psnr'])

        # Calculate PSNR, SSIM, and L2 between the adv images and the source image
        input_metrics = self.calculate_input_metrics(src_img, adv_imgs)
        self.input_l2_distances.append(input_metrics['l2'])
        self.input_ssim_values.append(input_metrics['ssim'])
        self.input_psnr_values.append(input_metrics['psnr'])

        # Log batch-wise metrics to wandb
        if self.use_wandb:
            wandb.log({
                'batch_success_rate': success_rate / self.batch_size,
                'input_batch_l2': input_metrics['l2'].item(),
                'input_batch_ssim': input_metrics['ssim'].item(),
                'input_batch_psnr': input_metrics['psnr'].item(),
                'output_batch_l2': output_metrics['l2'].item(),
                'output_batch_ssim': output_metrics['ssim'].item(),
                'output_batch_psnr': output_metrics['psnr'].item()
            })
        else:
            print(f"BATCH METRICS\nSuccess: {success_rate}/{self.batch_size}\nNormalized L2 Dist: {output_metrics['l2']}\nSSIM: {output_metrics['ssim']}\nPSNR: {output_metrics['psnr']}\n")

    def global_metrics(self) -> None:
        """
        Calculate and print the global average of success rate, PSNR, SSIM, and L2 distance.
        """
        asr = self._calculate_global_average(self.success_rates) / self.batch_size
        input_l2 = self._calculate_global_average(self.input_l2_distances)
        input_ssim = self._calculate_global_average(self.input_ssim_values)
        input_psnr = self._calculate_global_average(self.input_psnr_values)
        output_l2 = self._calculate_global_average(self.output_l2_distances)
        output_ssim = self._calculate_global_average(self.output_ssim_values)
        output_psnr = self._calculate_global_average(self.output_psnr_values)

        # Log global metrics to wandb
        if self.use_wandb:
            wandb.log({
                'global_asr': asr,
                'output_global_l2': output_l2,
                'output_global_ssim': output_ssim,
                'output_global_psnr': output_psnr,
                'input_global_l2': input_l2,
                'input_global_ssim': input_ssim,
                'input_global_psnr': input_psnr
            })
        else:
            print(f"\nGLOBAL METRICS\nASR: {asr}\nNormalized L2 Dist: {output_l2}\nSSIM: {output_ssim}\nPSNR: {output_psnr}")

    def calculate_success_rate(self, target_img: torch.Tensor, adv_imgs: torch.Tensor, model: torch.nn.Module) -> int:
        """
        Calculate the success rate of the attack by measuring the hamming distance between the compressed
        versions of target_img and adv_imgs.
        """
        success = 0
        with torch.no_grad():
            target_bytes = model.compress(target_img)['strings'][0][0]
            adv_bytes_list = model.compress(adv_imgs)['strings'][0]
        for adv_bytes in adv_bytes_list:
            hamming_dist = 0
            if len(adv_bytes) != len(target_bytes):
                continue
            for x_byte, adv_byte in zip(target_bytes, adv_bytes):
                hamming_dist += bin(x_byte ^ adv_byte).count('1')
            if hamming_dist == 0:
                success += 1
        return success

    def _calculate_global_average(self, metric: List[float]) -> float:
        return sum(metric) / len(metric) if metric else 0.0