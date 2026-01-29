import warnings

import torch.utils
warnings.filterwarnings("ignore")

import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

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
            if self.config['algorithm'] == 'mgd' or self.config['algorithm'] == 'hyp':
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
    
class MGDHyperprior(Attack):
    def criterion(self, src_emb, target_emb, src_hyp, target_hyp):
        return F.mse_loss(src_emb, target_emb) + F.mse_loss(src_hyp, target_hyp)
    
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
        target_output = self.model(target_img)
        target_emb, target_hyp = target_output['y_hat'], target_output['z']

        # Setup the optimizer and LR scheuler
        optimizer = torch.optim.Adam([src_imgs], lr=self.config['lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_steps'] / 10)
        
        # Track the best performance
        best_img = None #torch.rand_like(src_imgs)
        best_sr = 0
        loss_tracker = StatsMeter()

        pbar = tqdm(range(num_steps))
        for iter in pbar:  
            out = self.model(src_imgs)
            src_emb = out['y_hat']
            src_hyp = out['z']
            loss = self.criterion(src_emb, target_emb, src_hyp, target_hyp)
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
    
class CollisionAttack(Attack):
    """
    Collision-seeking attack aligned with myattack2.py:
    - First batch is targets; cache rounded embeddings + bitstreams.
    - Dot-grid mask always applied.
    - Hamming check happens before any optimizer step.
    - Constant LR (no scheduler decay).
    """
    def __init__(self, model, config, device):
        super().__init__(model, config, device)

    def _grid_grad_mask(self, image: torch.Tensor) -> torch.Tensor:
        v_skip = self.config.get('mgd', {}).get('vertical_skip', 4)
        h_skip = self.config.get('mgd', {}).get('horizontal_skip', 4)
        h, w = image.shape[-2:]
        mask = torch.ones((1, 1, h, w), device=self.device)
        mask[:, :, 0:h:v_skip, 0:w:h_skip] = 0.0
        return mask

    def _prepare_targets(self, tgt: torch.Tensor):
        with torch.no_grad():
            bytes_tgt = self.model.compress(tgt)['strings'][0]
            tgt_out = self.model.compress_till_rounding(tgt)
            emb_tgt = tgt_out['y_hat']
            emb_tgt_round = [torch.round(item) for item in emb_tgt]
            num_emb = sum(item[0].numel() for item in emb_tgt_round)
        return bytes_tgt, emb_tgt_round, num_emb

    def _embedding_loss(self, emb_adv, emb_tgt_round, num_emb: int) -> torch.Tensor:
        b = emb_tgt_round[0].shape[0]
        per_img = torch.zeros((b,), dtype=torch.float32, device=self.device)
        for adv_item, tgt_item in zip(emb_adv, emb_tgt_round):
            per_img += torch.sum(torch.square(adv_item - tgt_item), dim=tuple(range(1, adv_item.dim())))
        return torch.sum(per_img) / num_emb

    def _hamming_dist(self, bytes_tgt_list, bytes_adv_list) -> np.ndarray:
        hamm = []
        for tgt_bytes, adv_bytes in zip(bytes_tgt_list, bytes_adv_list):
            if len(tgt_bytes) != len(adv_bytes):
                hamm.append(len(tgt_bytes) * 8)
                continue
            dist = 0
            for t, a in zip(tgt_bytes, adv_bytes):
                dist += bin(t ^ a).count('1')
            hamm.append(dist)
        return np.array(hamm)

    def _save_collision_plot(
            self,
            x_src: torch.Tensor,
            x_adv: torch.Tensor,
            x_tgt: torch.Tensor,
            bytes_tgt,
            collision_idx: np.ndarray
        ):
        collision_idx = collision_idx.tolist()
        if len(collision_idx) == 0:
            return None

        with torch.no_grad():
            bytes_adv = self.model.compress(x_adv)['strings'][0]
            bytes_src = self.model.compress(x_src)['strings'][0]

        l2dist = torch.mean(torch.square(x_adv[collision_idx] - x_tgt[collision_idx]), dim=(1, 2, 3))
        best_pos = collision_idx[int(torch.argmax(l2dist))]

        output_dir = Path(__file__).resolve().parent.parent / "../results"
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / f'collision_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'

        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(x_src[best_pos].permute(1, 2, 0).detach().cpu())
        plt.axis('off')
        plt.title('src:' + bytes_src[best_pos].hex()[:10] + '...')

        plt.subplot(1, 3, 2)
        plt.imshow(x_adv[best_pos].permute(1, 2, 0).detach().cpu())
        plt.axis('off')
        plt.title('adv:' + bytes_adv[best_pos].hex()[:10] + '...')

        plt.subplot(1, 3, 3)
        plt.imshow(x_tgt[best_pos].permute(1, 2, 0).detach().cpu())
        plt.axis('off')
        plt.title('tgt:' + bytes_tgt[best_pos].hex()[:10] + '...')

        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()
        return fig_path

    def attack(self, x_target, dataloader, net, device):
        x_tgt = None
        bytes_tgt = emb_tgt_round = grad_mask = None
        global_loss = StatsMeter()
        best_adv = best_src = None

        for batch_idx, (x_src, _) in enumerate(dataloader):
            x_src = x_src.to(self.device)
            if batch_idx == 0:
                x_tgt = x_src.clone()
                bytes_tgt, emb_tgt_round, num_emb = self._prepare_targets(x_tgt)
                grad_mask = self._grid_grad_mask(x_tgt)  # always apply dot grid
                continue

            x_adv, batch_loss, hamm = self.attack_batch(
                x_tgt, x_src, emb_tgt_round, bytes_tgt, num_emb, mask=grad_mask
            )
            for v in batch_loss.data:
                global_loss.update(v)

            self.evaluator.batch_eval(x_src, x_tgt, x_adv)
            best_adv, best_src = x_adv, x_src

            collision_idx = np.where(hamm == 0)[0]
            if len(collision_idx) > 0:
                fig_path = self._save_collision_plot(x_src, x_adv, x_tgt, bytes_tgt, collision_idx)
                if fig_path:
                    print(f"Collision Detected. Saved subplot to {fig_path}")
                

            if batch_idx >= self.config['num_batches'] - 1:
                print("Reached configured number of batches without collision.")
                break

        self.evaluator.global_eval()
        return x_tgt, best_adv, best_src, global_loss

    def attack_batch(
            self,
            target_img: torch.Tensor,
            src_imgs: torch.Tensor,
            emb_tgt_round,
            bytes_tgt,
            num_emb: int,
            mask: torch.Tensor = None
        ) -> Tuple:
        src_imgs = src_imgs.to(self.device)
        x_adv = src_imgs.clone()
        x_adv.requires_grad = True

        optimizer = torch.optim.Adam([x_adv], lr=self.config['lr'])
        loss_tracker = StatsMeter()
        hamm = np.array([])

        pbar = tqdm(range(self.config['num_steps']))
        for _ in pbar:
            out = self.model.compress_till_rounding(x_adv)
            emb_adv = out['y_hat']

            # Pre-step collision check (matches myattack2 flow)
            max_round_diff = torch.zeros((src_imgs.shape[0],), device=self.device)
            for tgt_item, adv_item in zip(emb_tgt_round, emb_adv):
                max_round_diff = torch.maximum(
                    max_round_diff,
                    torch.amax(torch.abs(torch.round(adv_item) - tgt_item), dim=tuple(range(1, adv_item.dim())))
                )
            if torch.min(max_round_diff) <= 2:
                bytes_adv = self.model.compress(x_adv)['strings'][0]
                hamm = self._hamming_dist(bytes_tgt, bytes_adv)
                if np.any(hamm == 0):
                    break

            loss = self._embedding_loss(emb_adv, emb_tgt_round, num_emb)
            pbar.set_description(f"[Running collision attack]: Loss {loss.item():.6f}")
            if wandb_available:
                wandb.log({
                    f'Batch {self.batch_num} Loss': loss
                })

            optimizer.zero_grad()
            x_adv.grad, = torch.autograd.grad(loss, [x_adv])
            if mask is not None:
                x_adv.grad *= mask
            optimizer.step()
            x_adv.data = torch.clamp(x_adv.data, 0, 1)
            loss_tracker.update(loss.item())

        return x_adv.detach(), loss_tracker, hamm
    
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
