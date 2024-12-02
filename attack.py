import warnings

import torch.utils
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import List, Dict, Tuple
from tqdm import tqdm
 
from utils import StatsMeter
from masks import ring_mask, box_mask, dot_mask
from evaluator import MetricsEvaluator
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
            config: Dict,
            device: str
        ):
        self.model = model
        self.batch_size = 24 if config['dataset'] == 'kodak' else config['batch_size']
        self.device = device
        self.batch_num = 0
        self.config = config

        # Evaluation metrics
        self.evaluator = MetricsEvaluator(self.batch_size, use_wandb=wandb_available)
        
    def _init_mask(
            self,
            image: torch.Tensor,
            mask_type: str = 'dot',
        ) -> torch.Tensor:
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
    
    def _apply_gradient_mask(self, grad: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
    
    def attack(
            self, 
            x: torch.Tensor, 
            dataloader: torch.utils.data.DataLoader, 
            net: torch.nn.Module, 
            device: torch.device
            ) -> Tuple:
        x_clone = x.clone()
        for i, (x_hat, _) in enumerate(dataloader):
            x_hat = x_hat.to(device)
            x_src = x_hat.clone()
            x_hat.requires_grad = True
            mask = self._init_mask(x_hat, self.config['mask_type']) if self.config['algorithm'] == 'mgd' else None
            x_adv, loss_tracker = self.attack_batch(x, x_hat, mask=mask)
            with torch.no_grad():
                output = net(x_adv)['x_hat']

            self.batch_eval(x_clone, x_adv, output)
            if i == self.config['num_batches'] - 1:
                break

        self.global_eval()
        return (x, x_adv, x_src, loss_tracker)
    
    def attack_batch(self):
        raise NotImplementedError
    
    def batch_eval(
            self, 
            x: torch.Tensor, 
            x_adv: torch.Tensor,
            output: torch.Tensor,
        ) -> None:
        self.evaluator.evaluate_batch(x, x_adv, output, self.model)

    def global_eval(self) -> None:
        self.evaluator.global_metrics()

class MGD(Attack):
    def attack_batch(
            self,
            src_img: torch.Tensor, 
            target_img: torch.Tensor,
            mask: torch.Tensor = None
        ) -> Tuple:
        # Count the number of batches that we have attacked
        self.batch_num += 1
        num_steps = self.config['num_steps']

        # Move images to the same device as the model
        src_img = src_img.to(self.device)
        target_img = target_img.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        # Get the embedding of the source image and make a copy of the target
        src_emb = self.model(src_img)['y_hat']

        # Setup the optimizer and LR scheuler
        optimizer = torch.optim.Adam([target_img], lr=self.config['lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_steps'] / 10)
        
        # Track the best performance
        best_img = None
        best_sr = 0
        loss_tracker = StatsMeter()

        pbar = tqdm(range(num_steps))
        for iter in pbar:  
            out = self.model(target_img)
            target_emb = out['y_hat']
            loss = self.criterion(src_emb, target_emb)
            pbar.set_description(f"[Running attack]: Loss {loss.item()}")
            optimizer.zero_grad()
            target_img.grad, = torch.autograd.grad(loss, [target_img])
            target_img.grad = self._apply_gradient_mask(target_img.grad, mask)
            optimizer.step()
            scheduler.step()
            loss_tracker.update(loss.item())
            if wandb_available:
                wandb.log({
                    f'Batch {self.batch_num} Loss': loss
                })
                
            # Save the image that achieved the best performance
            if iter % 100 == 0:
                s = self.evaluator.calculate_success_rate(src_img, target_img, self.model) / self.batch_size
                if s >= best_sr:
                    best_sr = s
                    best_img = target_img.clone()

        return best_img, loss_tracker
    
class PGD(Attack):
    def __init__(self, model, config, device):
        super().__init__(model, config, device)
        self.eta = config['pgd']['eta']
        
    def attack_batch(
            self,
            src_img : torch.Tensor, 
            target_img : torch.Tensor, 
            mask : torch.Tensor = None
        ) -> Tuple:
        # Count the number of batches that we have attacked
        self.batch_num += 1
        num_steps = self.config['num_steps']

        # Move images to the same device as the model
        src_img = src_img.to(self.device)
        target_img = target_img.to(self.device)
        attack_img = target_img.clone().to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        # Get the embedding of the source image and make a copy of the target
        src_emb = self.model(src_img)['y_hat']

        # Setup the optimizer and LR scheuler
        optimizer = torch.optim.SGD([target_img], lr=self.config['lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_steps'] / 10)

        # Track the best performance
        best_img = None
        best_sr = 0
        loss_tracker = StatsMeter()

        pbar = tqdm(range(num_steps))
        for iter in pbar:  
            out = self.model(target_img)
            target_emb = out['y_hat']
            loss = self.criterion(src_emb, target_emb)
            pbar.set_description(f"[Running attack]: Loss {loss.item()}")
            optimizer.zero_grad()
            target_img.grad, = torch.autograd.grad(loss, [target_img])
            target_img.grad = target_img.grad.sign()    # Use the sign of the gradients
            optimizer.step()
            scheduler.step()
            # Project perturbation onto the eta ball
            with torch.no_grad():
                perturbation = target_img - attack_img
                perturbation = torch.clamp(perturbation, -self.eta, self.eta)
                target_img.data = attack_img.data + perturbation.data
            loss_tracker.update(loss.item())
            if wandb_available:
                wandb.log({
                    f'Batch {self.batch_num} Loss': loss
                })
                
            if iter % 100 == 0:
                s = self.evaluator.calculate_success_rate(src_img, target_img, self.model) / self.batch_size
                if s >= best_sr:
                    best_sr = s
                    best_img = target_img.clone()

        return best_img, loss_tracker

class CW(Attack):
    def __init__(self, model, config, device):
        super().__init__(model, config, device)
        self.c = config['cw']['c']
    
    def criterion(
            self,
            src_x: torch.Tensor, 
            target_x: torch.Tensor,
            target_img: torch.Tensor, 
            adv_img: torch.Tensor
        ) -> torch.Tensor:
        loss = F.mse_loss(src_x, target_x) + self.c * F.mse_loss(target_img, adv_img)
        return loss
    
    def tanh_space(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x: torch.Tensor) -> torch.Tensor:
        return torch.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

    def attack_batch(
            self,
            src_img : torch.Tensor, 
            target_img : torch.Tensor,
            mask : torch.Tensor = None
        ) -> Tuple:
        # Count the number of batches that we have attacked
        self.batch_num += 1
        num_steps = self.config['num_steps']

        # Move images to the same device as the model
        src_img = src_img.to(self.device)
        target_img = target_img.to(self.device)
        attack_img = target_img.clone().to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        # Project the image into inv tanh space
        w = self.inverse_tanh_space(target_img).detach().requires_grad_()

        # Setup the optimizer and LR scheuler
        optimizer = torch.optim.Adam([w], lr=self.config['lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_steps'] / 10)

        # Get the embedding of the source image and make a copy of the target
        src_emb = self.model(src_img)['y_hat']

        # Track the best performance
        best_img = None
        best_sr = 0
        loss_tracker = StatsMeter()

        pbar = tqdm(range(num_steps))
        for iter in pbar:  
            adv_imgs = self.tanh_space(w)
            out = self.model(adv_imgs)
            target_emb = out['y_hat']
            loss = self.criterion(src_emb, target_emb, adv_imgs, attack_img)
            pbar.set_description(f"[Running attack]: Loss {loss.item()}")
            optimizer.zero_grad()
            w.grad, = torch.autograd.grad(loss, [w])
            optimizer.step()
            scheduler.step()
            loss_tracker.update(loss.item())
            if wandb_available:
                wandb.log({
                    f'Batch {self.batch_num} Loss': loss
                })
                
            if iter % 100 == 0:
                s = self.evaluator.calculate_success_rate(src_img, adv_imgs, self.model) / self.batch_size
                if s >= best_sr:
                    best_sr = s
                    best_img = adv_imgs.clone()
        
        return best_img, loss_tracker