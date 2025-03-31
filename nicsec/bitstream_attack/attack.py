import warnings

import torch.utils
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import List, Dict, Tuple
from tqdm import tqdm
 
from .utils import StatsMeter
from .masks import ring_mask, box_mask, dot_mask
from .evaluator import Evaluator
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
        self.evaluator = Evaluator(self.config, self.model)
        
    def _init_mask(
            self,
            image: torch.Tensor,
            mask_type: str = 'dot',
        ) -> torch.Tensor:
        if mask_type == 'box':
            mask = box_mask()
        elif mask_type == 'ring':
            num_rings = 100
            ring_width = 1
            ring_separation = 2
            mask = ring_mask(image, num_rings=num_rings, ring_width=ring_width, ring_separation=ring_separation)
        elif mask_type == 'dot':
            vertical_skip = self.config['mgd']['vertical_skip'] 
            horizontal_skip = self.config['mgd']['horizontal_skip'] 
            mask = dot_mask(image, vertical_skip=vertical_skip, horizontal_skip=horizontal_skip)
        elif mask_type == 'learned':
            # Initialize the mask as a learnable parameter
            vertical_skip = self.config['mgd']['vertical_skip'] 
            horizontal_skip = self.config['mgd']['horizontal_skip'] 
            mask = dot_mask(image, vertical_skip=vertical_skip, horizontal_skip=horizontal_skip)
            mask = torch.nn.Parameter(torch.tensor(mask.detach().cpu().numpy()).detach().to(self.device)) # convert tensor into leaf node
        return mask
    
    def _apply_gradient_mask(self, grad: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask is not None:
            grad *= mask
        return grad
    
    def criterion(
            self,
            src_emb: torch.Tensor, 
            target_emb: torch.Tensor,
        ) -> torch.Tensor:
        return F.mse_loss(src_emb, target_emb) 
    
    def attack(
            self, 
            x_target: torch.Tensor, 
            dataloader: torch.utils.data.DataLoader, 
            net: torch.nn.Module, 
            device: torch.device
            ) -> Tuple:
        for i, (x_hat, _) in enumerate(dataloader):
            x_hat = x_hat.to(device)
            x_src = x_hat.clone()
            x_hat.requires_grad = True
            if self.config['algorithm'] == 'mgd' or self.config['algorithm'] == 'cw':
                mask = self._init_mask(x_hat, self.config['mask_type']) 
            else:
                mask = None

            x_adv, loss_tracker = self.attack_batch(x_target, x_hat, mask=mask)
            self.evaluator.batch_eval(x_src, x_target, x_adv)
            if i == self.config['num_batches'] - 1:
                break

        self.evaluator.global_eval()
        return (x_target, x_adv, x_src, loss_tracker)
    
    def attack_batch(self):
        raise NotImplementedError

class MGD(Attack):
    def attack_batch(
            self,
            target_img: torch.Tensor, 
            src_imgs: torch.Tensor,
            mask: torch.Tensor = None
        ) -> Tuple:
        # Count the number of batches that we have attacked
        self.batch_num += 1
        num_steps = self.config['num_steps']

        # Move images to the same device as the model
        target_img = target_img.to(self.device)
        src_imgs = src_imgs.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        # Get the embedding of the target image
        target_emb = self.model(target_img)['y_hat']

        # Setup the optimizer and LR scheuler
        optimizer = torch.optim.Adam([src_imgs], lr=self.config['lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_steps'] / 10)
        
        # Track the best performance
        best_img = None
        best_sr = 0
        loss_tracker = StatsMeter()

        pbar = tqdm(range(num_steps))
        for iter in pbar:  
            out = self.model(src_imgs)
            src_emb = out['y_hat']
            loss = self.criterion(target_emb, src_emb)
            pbar.set_description(f"[Running attack]: Loss {loss.item()}")
            optimizer.zero_grad()
            src_imgs.grad, = torch.autograd.grad(loss, [src_imgs])
            src_imgs.grad = self._apply_gradient_mask(src_imgs.grad, mask)
            optimizer.step()
            scheduler.step()
            
            loss_tracker.update(loss.item())
            if wandb_available:
                wandb.log({
                    f'Batch {self.batch_num} Loss': loss
                })
                
            # Save the images that achieved the best performance
            if iter % 100 == 0:
                s = self.evaluator._asr(target_img, src_imgs)
                if s >= best_sr:
                    best_sr = s
                    best_img = src_imgs.clone()

        return best_img, loss_tracker
    
class MGD2(Attack):
    def criterion(self, src_emb, target_emb, ord='inf'):
        if ord == 'inf':
            return torch.dist(src_emb, target_emb, p=float('inf'))
        else:
            return F.mse_loss(src_emb, target_emb)

    def attack_batch(
            self,
            target_img: torch.Tensor, 
            src_imgs: torch.Tensor,
            mask: torch.Tensor = None
        ) -> Tuple:
        # Count the number of batches that we have attacked
        self.batch_num += 1
        num_steps = self.config['num_steps']

        # Move images to the same device as the model
        target_img = target_img.to(self.device)
        src_imgs = src_imgs.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        # Get the embedding of the target image
        target_emb = self.model(target_img)['y_hat']

        # Setup the optimizer and LR scheuler
        optimizer = torch.optim.Adam([src_imgs], lr=self.config['lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_steps'] / 10)
        
        # Track the best performance
        best_img = None
        best_sr = 0
        loss_tracker = StatsMeter()

        pbar = tqdm(range(num_steps))
        for iter in pbar:
            out = self.model(src_imgs)
            src_emb = out['y_hat']
            loss = self.criterion(target_emb, src_emb, ord='2' if iter <= (num_steps//2) else 'inf')
            pbar.set_description(f"[Running attack]: Loss {loss.item()}")
            optimizer.zero_grad()
            src_imgs.grad, = torch.autograd.grad(loss, [src_imgs])
            src_imgs.grad = self._apply_gradient_mask(src_imgs.grad, mask)
            optimizer.step()
            scheduler.step()

            # Update tracking
            loss_tracker.update(loss.item())

            # Save the best adv image
            if iter % 100 == 0:
                s = self.evaluator._asr(target_img, src_imgs)
                if s >= best_sr:
                    best_sr = s
                    best_img = src_imgs.clone()

        return best_img, loss_tracker
    
class PGD(Attack):
    def __init__(self, model, config, device):
        super().__init__(model, config, device)
        self.eta = config['pgd']['eta']
        
    def attack_batch(
            self,
            target_img : torch.Tensor, 
            src_imgs : torch.Tensor, 
            mask : torch.Tensor = None
        ) -> Tuple:
        # Count the number of batches that we have attacked
        self.batch_num += 1
        num_steps = self.config['num_steps']

        # Move images to the same device as the model
        target_img = target_img.to(self.device)
        src_imgs = src_imgs.to(self.device)
        attack_img = src_imgs.clone().to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        # Get the embedding of the source image and make a copy of the target
        target_emb = self.model(target_img)['y_hat']

        # Setup the optimizer and LR scheuler
        optimizer = torch.optim.SGD([src_imgs], lr=self.config['lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_steps'] / 10)

        # Track the best performance
        best_img = None
        best_sr = 0
        loss_tracker = StatsMeter()

        pbar = tqdm(range(num_steps))
        for iter in pbar:  
            out = self.model(src_imgs)
            src_emb = out['y_hat']
            loss = self.criterion(src_emb, target_emb)
            pbar.set_description(f"[Running attack]: Loss {loss.item()}")
            optimizer.zero_grad()
            src_imgs.grad, = torch.autograd.grad(loss, [src_imgs])
            src_imgs.grad = src_imgs.grad.sign()    # Use the sign of the gradients
            optimizer.step()
            scheduler.step()
            # Project perturbation onto the eta ball
            with torch.no_grad():
                perturbation = src_imgs - attack_img
                perturbation = torch.clamp(perturbation, -self.eta, self.eta)
                src_imgs.data = attack_img.data + perturbation.data
            loss_tracker.update(loss.item())
            if wandb_available:
                wandb.log({
                    f'Batch {self.batch_num} Loss': loss
                })
                
            if iter % 100 == 0:
                s = self.evaluator._asr(target_img, src_imgs)
                if s >= best_sr:
                    best_sr = s
                    best_img = src_imgs.clone()

        return best_img, loss_tracker
    
class CW(Attack):
    def __init__(self, model, config, device):
        super().__init__(model, config, device)
        self.c = config['cw']['c']
    
    def criterion(
            self,
            target_emb: torch.Tensor, 
            adv_emb: torch.Tensor,
            adv_img: torch.Tensor, 
            src_img: torch.Tensor
        ) -> torch.Tensor:
        loss = F.mse_loss(target_emb, adv_emb) + self.c * F.mse_loss(src_img, adv_img)
        return loss
    
    def tanh_space(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x: torch.Tensor) -> torch.Tensor:
        return torch.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

    def attack_batch(
            self,
            target_img : torch.Tensor, 
            src_imgs : torch.Tensor,
            mask : torch.Tensor = None
        ) -> Tuple:
        # Count the number of batches that we have attacked
        self.batch_num += 1
        num_steps = self.config['num_steps']

        # Move images to the same device as the model
        target_img = target_img.to(self.device)
        src_imgs = src_imgs.to(self.device)
        attack_img = src_imgs.clone().to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        # Project the image into inv tanh space
        w = self.inverse_tanh_space(src_imgs).detach().requires_grad_()

        # Setup the optimizer and LR scheuler
        optimizer = torch.optim.Adam([w], lr=self.config['lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_steps'] / 10)

        # Get the embedding of the source image 
        target_emb = self.model(target_img)['y_hat']

        # Track the best performance
        best_img = None
        best_sr = 0
        loss_tracker = StatsMeter()

        pbar = tqdm(range(num_steps))
        for iter in pbar:  
            adv_imgs = self.tanh_space(w)
            out = self.model(adv_imgs)
            adv_emb = out['y_hat']
            loss = self.criterion(target_emb, adv_emb, adv_imgs, attack_img)
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
                s = self.evaluator._asr(target_img, src_imgs)
                if s >= best_sr:
                    best_sr = s
                    best_img = adv_imgs.clone()
        
        return best_img, loss_tracker