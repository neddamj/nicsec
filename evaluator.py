import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from sewar.full_ref import msssim
import numpy as np
from typing import List, Dict
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb = None
    wandb_available = False

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from sewar.full_ref import msssim
import numpy as np
from typing import List, Dict
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb = None
    wandb_available = False

class Evaluator:
    def __init__(self, config, compressor):
        self.config = config
        self.compressor = compressor
        self.metrics = {
            "msssim_target_adv": [],
            "msssim_src_adv": [],
            "l2_target_adv": [],
            "l2_src_adv": [],
            "asr": []
        }

    def _msssim(self,
                img1: torch.Tensor,
                img2: torch.Tensor):
        """
        Compute the average MS-SSIM between pairs of images.
        If img1 is a single reference image and img2 a batch of images,
        then for each i we compare img1 (or img1[0]) with img2[i].  
        Here we assume that both tensors have the same batch dimension.
        """
        avg_msssim = []
        # If the two inputs have the same batch size, compare corresponding images.
        if img1.shape[0] == img2.shape[0]:
            for i in range(img1.shape[0]):
                # Convert tensor image from [C, H, W] to numpy array [H, W, C]
                img1_np = img1[i].permute(1, 2, 0).detach().cpu().numpy()
                img2_np = img2[i].permute(1, 2, 0).detach().cpu().numpy()

                # Convert to uint8 if images are in range [0,1]
                img1_np = (img1_np * 255).astype(np.uint8)
                img2_np = (img2_np * 255).astype(np.uint8)

                msssim_score = msssim(img1_np, img2_np).astype(np.float64)
                avg_msssim.append(msssim_score)
        else:
            # If img1 is a single image (or only one image is intended to be the reference)
            # then use img1[0] for every comparison.
            ref = img1[0]
            for i in range(img2.shape[0]):
                ref_np = ref.permute(1, 2, 0).detach().cpu().numpy()
                img2_np = img2[i].permute(1, 2, 0).detach().cpu().numpy()

                ref_np = (ref_np * 255).astype(np.uint8)
                img2_np = (img2_np * 255).astype(np.uint8)

                msssim_score = msssim(ref_np, img2_np).astype(np.float64)
                avg_msssim.append(msssim_score)
                
        return torch.tensor(avg_msssim).mean()
        
    def _l2(self,
            img1: torch.Tensor,
            img2: torch.Tensor):
        """
        Compute the average per-sample L2 (Euclidean) distance normalized by the number of pixels.
        Assumes img1 and img2 are batches of images of shape [B, C, H, W].
        """
        diff = (img1 - img2).view(img1.shape[0], -1)
        norms = diff.norm(p=2, dim=1) / np.sqrt(diff.shape[1])
        return norms.mean()
    
    def _hamming_dist(self,
                      target_img: torch.Tensor,
                      adv_imgs: torch.Tensor):
        """
        Compute the hamming distance between the compressed representations.
        This function compresses the target image and each adversarial image, and computes
        the bit-level differences between the resulting byte strings.
        """
        with torch.no_grad():
            # Compress the target image; assuming target_img is a single image or a batch with one element.
            target_bytes = self.compressor.compress(target_img)['strings'][0][0]
            adv_bytes_list = self.compressor.compress(adv_imgs)['strings'][0]

        hamm = []   # list to store the hamming distance for each adv image
        success = 0 # counter for cases with zero hamming distance
        for adv_bytes in adv_bytes_list:
            # Only compare if lengths match.
            if len(adv_bytes) != len(target_bytes):
                continue
            hamming_dist = 0
            for target_byte, adv_byte in zip(target_bytes, adv_bytes):
                # XOR the bytes and count the number of set bits.
                hamming_dist += bin(target_byte ^ adv_byte).count('1')
            hamm.append(hamming_dist)
            if hamming_dist == 0:
                success += 1
        return torch.tensor(hamm), success
            
    def _asr(self,
             target_img: torch.Tensor,
             adv_imgs: torch.Tensor):
        """
        Compute the attack success rate (ASR) as the fraction of adversarial images whose 
        compressed representations are identical (i.e. hamming distance zero).
        """
        _, success = self._hamming_dist(target_img, adv_imgs)
        # Here we use the number of adversarial images as denominator.
        return success / adv_imgs.shape[0]

    def batch_eval(self, 
                   src_imgs: torch.Tensor, 
                   target_img: torch.Tensor, 
                   adv_imgs: torch.Tensor):
        """
        Evaluate the batch of images using multiple metrics.
        """
        # Compute MS-SSIM between target and adversarial images.
        msssim_target_adv = self._msssim(target_img, adv_imgs)
        # Compute MS-SSIM between source and adversarial images.
        msssim_src_adv = self._msssim(src_imgs, adv_imgs)

        # Compute normalized L2 distances.
        l2_target_adv = self._l2(target_img, adv_imgs)
        l2_src_adv = self._l2(src_imgs, adv_imgs)

        # Compute attack success rate.
        asr = self._asr(target_img, adv_imgs)

        # Store batch metrics for global computation.
        self.metrics["msssim_target_adv"].append(msssim_target_adv.item())
        self.metrics["msssim_src_adv"].append(msssim_src_adv.item())
        self.metrics["l2_target_adv"].append(l2_target_adv.item())
        self.metrics["l2_src_adv"].append(l2_src_adv.item())
        self.metrics["asr"].append(asr)

        if wandb_available:
            wandb.log({
                'batch_success_rate': asr,
                'msssim_target_adv': msssim_target_adv.item(),
                'msssim_src_adv': msssim_src_adv.item(),
                'l2_target_adv': l2_target_adv.item(),
                'l2_src_adv': l2_src_adv.item()
            })

    def global_eval(self):
        """
        Compute global (average) results over all batches.
        """
        global_results = {key: np.mean(values) for key, values in self.metrics.items()}

        if wandb_available:
            wandb.log({
                'global_success_rate': global_results["asr"],
                'global_msssim_target_adv': global_results["msssim_target_adv"],
                'global_msssim_src_adv': global_results["msssim_src_adv"],
                'global_l2_target_adv': global_results["l2_target_adv"],
                'global_l2_src_adv': global_results["l2_src_adv"]
            })
        return global_results
