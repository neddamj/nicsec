import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from typing import List, Dict
from tqdm import tqdm
 
from utils import StatsMeter
from masks import ring_mask, box_mask, dot_mask
from eval import MetricsEvaluator
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb = None
    wandb_available = False
    
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
        
    def _init_mask(
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
            horizontal_skip = 2
            mask = dot_mask(image, vertical_skip=vertical_skip, horizontal_skip=horizontal_skip)
        return mask
    
    def apply_gradient_mask(self, grad: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask is not None:
            grad *= mask
        return grad
    
    def _init_scheduler(
            self,
            config: Dict,
            optimizer: torch.optim.Optimizer,
            scale_factor : float = 0.1
    ) -> torch.optim.lr_scheduler.LRScheduler:
        scheduler_type = config['scheduler_type']
        if scheduler_type == 'lambda':
            lambda_lr = lambda iteration: scale_factor ** (iteration // (config['num_steps']/2))
            scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)
        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=config['num_steps'] / 10)
        else:
            raise ValueError("Invalid scheduler type. Use 'cosine' or 'lambda'.")
        return scheduler
    
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
        x_clone = x.clone()
        for i, (x_hat, _) in enumerate(dataloader):
            x_hat = x_hat.to(device)
            x_src = x_hat.clone()
            x_hat.requires_grad = True
            mask = self._init_mask(x_hat, config['mask_type'])

            optimizer = torch.optim.Adam([x_hat], lr=config['lr'])
            scheduler = self._init_scheduler(config, optimizer)
            #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_steps'] / 10)
            x_adv, loss_tracker = self.attack_batch(x, x_hat, optimizer=optimizer, scheduler=scheduler, num_steps=config['num_steps'], mask=mask)
            with torch.no_grad():
                output = net(x_adv)['x_hat']

            self.batch_eval(x_clone, x_adv, output)
            if i == config['num_batches'] - 1:
                break

        self.global_eval()
        return (x, x_adv, loss_tracker)
    
    def batch_eval(
            self, 
            x: torch.Tensor, 
            x_adv: torch.Tensor,
            output: torch.Tensor,
        ) -> None:
        self.evaluator.evaluate_batch(x, x_adv, output, self.model)

    def global_eval(self) -> None:
        self.evaluator.global_metrics()