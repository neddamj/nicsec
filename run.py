import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb = None
    wandb_available = False

from attack import Attack
from compressor import NeuralCompressor, JpegCompressor

# Disable TF32 Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda' 
elif torch.backends.mps.is_available():
    device = 'mps'

def get_compressor(compressor_type, model_id, device, quality_factor=2, image_size=256):
    if compressor_type == 'neural':
        return NeuralCompressor(model_id=model_id, device=device)
    elif compressor_type == 'jpeg':
        return JpegCompressor(differentiable=True, quality_factor=quality_factor, image_size=image_size, device=device)
    else:
        raise ValueError("Invalid compressor type. Use 'neural' or 'jpeg'.")

def setup_wandb(config, wandb_available):
    # Setup wandb logging
    if wandb_available:
        wandb.init(project="neural-image-compression-attack")
        wandb.config.update(config)

def main(config):
    compressor = get_compressor(config['compressor_type'], config['model_id'], device, config['image_size'])

    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])), 
        transforms.ToTensor()
        ])
    dataset = datasets.CelebA(root='./data', split='train', transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    setup_wandb(config, wandb_available)

    x = dataset[0][0].unsqueeze(0).to(device)
    attack = Attack(model=compressor, batch_size=config['batch_size'], device=device)
    x_src, x_adv, loss_tracker =  attack.attack(x, dataloader, compressor, device, config)

if __name__ == '__main__':
    # Run the attack
    config = {
        'lr': 3e-2,
        'batch_size': 2,
        'num_batches': 1,
        'num_steps': 1000,
        'image_size': 256,
        'mask_type': 'dot', 
        'compressor_type': 'jpeg',   
        'model_id': 'my_bmshj2018_factorized_relu'
    }
    main(config)