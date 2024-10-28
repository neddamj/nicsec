import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from tqdm import tqdm
from pytorch_msssim import ssim

from utils import StatsMeter
from typing import List, Dict
from masks import ring_mask, box_mask, dot_mask

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
        return ssim(img1, img2.unsqueeze(0), data_range=1.0)

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
        success_rate = self._calculate_success_rate(src_img, adv_imgs, model)
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

    def _calculate_success_rate(self, src_img: torch.Tensor, adv_imgs: torch.Tensor, model: torch.nn.Module) -> int:
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
    
class Attack:
    def __init__(
            self, 
            model: torch.nn.Module, 
            batch_size: int,
            device: str
        ):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.batch_num = 0

        # Evaluation metrics
        self.evaluator = MetricsEvaluator(self.batch_size, use_wandb=wandb_available)
        
    def init_mask(
            self,
            image: torch.Tensor,
            mask_type: str = 'dot',
        ) -> None:
        if mask_type == 'box':
            mask = box_mask()
        elif mask_type == 'ring':
            num_rings = 50
            ring_width = 1
            ring_separation = 5
            mask = ring_mask(image, num_rings=num_rings, ring_width=ring_width, ring_separation=ring_separation)
        elif mask_type == 'dot':
            vertical_skip = 2 
            horizontal_skip = 3
            mask = dot_mask(image, vertical_skip=vertical_skip, horizontal_skip=horizontal_skip)
        return mask
    
    def apply_gradient_mask(self, grad: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask is not None:
            grad *= mask
        return grad
    
    def criterion(
            self,
            src_img: torch.Tensor, 
            target_img: torch.Tensor
        ) -> torch.Tensor:
        mse = F.mse_loss(src_img, target_img) 
        cos_sim = F.cosine_similarity(src_img.view(1, -1), target_img.view(self.batch_size, -1), dim=1).mean()
        loss = mse + ((1 - cos_sim) / 2)
        return loss

    def attack_batch(
            self,
            src_img: torch.Tensor, 
            target_img: torch.Tensor, 
            optimizer: torch.optim.Optimizer, 
            scheduler: torch.optim.lr_scheduler._LRScheduler, 
            num_steps: int, 
            mask: torch.Tensor = None
        ) -> List:
        # Count the number of batches that we have attacked
        self.batch_num += 1

        # Move images to the same device as the model
        src_img = src_img.to(self.device)
        target_img = target_img.to(self.device)
        mask = mask.to(self.device)
        # Get the embedding of the source image and make a copy of the target
        src_emb = self.model(src_img)['y_hat']

        # Track the best performance
        best_img = None
        loss_tracker = StatsMeter()

        pbar = tqdm(range(num_steps))
        for iter in pbar:  
            out = self.model(target_img)
            target_emb = out['y_hat']
            loss = self.criterion(src_emb, target_emb)
            pbar.set_description(f"[Running attack]: Loss {loss.item()}")
            optimizer.zero_grad()
            target_img.grad, = torch.autograd.grad(loss, [target_img])
            target_img.grad = self.apply_gradient_mask(target_img.grad, mask)
            optimizer.step()
            scheduler.step()
            loss_tracker.update(loss.item())
            if wandb_available:
                wandb.log({
                    f'Batch {self.batch_num} Loss': loss
                })
                
            # Save the image that achieved the best performance
            if loss.item() == loss_tracker.min:
                best_img = target_img

        return best_img, loss_tracker
    
    def attack(self, x, dataloader, net, device, config):
        for i, (x_hat, _) in enumerate(dataloader):
            x_hat = x_hat.to(device)
            x_src = x_hat.clone()
            x_hat.requires_grad = True
            mask = self.init_mask(x_hat, config['mask_type'])

            optimizer = torch.optim.Adam([x_hat], lr=config['lr'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_steps'] / 10)
            x_adv, loss_tracker = self.attack_batch(x, x_hat, optimizer=optimizer, scheduler=scheduler, num_steps=config['num_steps'], mask=mask)
            with torch.no_grad():
                output = net(x_adv)['x_hat']

            self.batch_eval(x, x_adv, output)
            if i == config['num_batches'] - 1:
                break

        self.global_eval()
        return (x_src, x_adv, loss_tracker)
    
    def batch_eval(
            self, 
            x: torch.Tensor, 
            x_adv: torch.Tensor,
            output: torch.Tensor,
        ) -> None:
        self.evaluator.evaluate_batch(x, x_adv, output, self.model)

    def global_eval(self) -> None:
        self.evaluator.global_metrics()