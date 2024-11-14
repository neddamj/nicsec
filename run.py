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

from attack import MGD, PGD, CW
from compressor import NeuralCompressor, JpegCompressor

# Disable TF32 Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:2'
elif torch.backends.mps.is_available():
    device = 'mps'

def get_compressor(config, device):
    if config['compressor_type'] == 'neural':
        return NeuralCompressor(model_id=config['model_id'], device=device)
    elif config['compressor_type'] == 'jpeg':
        return JpegCompressor(differentiable=True, quality_factor=config['quality_factor'], image_size=config['image_size'], device=device)
    else:
        raise ValueError("Invalid compressor type. Use 'neural' or 'jpeg'.")
    
def get_dataset(config, transform):
    if config['dataset'] == 'celeba':
        dataset = datasets.CelebA(root='./data', split='train', transform=transform, download=True)
    elif config['dataset'] == 'imagenette':
        dataset = datasets.Imagenette(root='./data', split='train', transform=transform)
    else:
        raise ValueError("Invalid dataset. Use 'celeba' or 'imagenette'.")
    return dataset

def get_attack_algo(config, compressor):
    if config['algorithm'] == 'mgd':
        attack = MGD(model=compressor, batch_size=config['batch_size'], device=device)
    elif config['algorithm'] == 'pgd':
        attack = PGD(model=compressor, batch_size=config['batch_size'], device=device, eta=config['pgd']['eta'])
    elif config['algorithm'] == 'cw':
        attack = CW(model=compressor, batch_size=config['batch_size'], device=device, c=config['cw']['c'])
    else:
        raise ValueError("Invalid algorithm. Use 'mgd' or 'pgd'.")
    return attack

def setup_wandb(config, wandb_available):
    # Setup wandb logging
    if wandb_available:
        wandb.init(project="neural-image-compression-attack")
        wandb.config.update(config)

def direct_attack(config):
    compressor = get_compressor(config, device)

    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])), 
        transforms.ToTensor()
        ])
    dataset = get_dataset(config, transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    setup_wandb(config, wandb_available)

    x = dataset[0][0].unsqueeze(0).to(device)
    attack = get_attack_algo(config, compressor)
    x_src, x_adv, x_target, loss_tracker =  attack.attack(x, dataloader, compressor, device, config)
    return x_src, x_adv, x_target, loss_tracker

if __name__ == '__main__':
    # Run the attack
    config = {
        'lr': 3e-2,
        'batch_size': 16,
        'num_batches': 1,
        'num_steps': 5000,
        'image_size': 256, 
        'quality_factor': 1,
        'mask_type': 'dot',
        'dataset': 'imagenette',        # 'celeba' or imagenette'
        'compressor_type': 'neural',      # 'neural' or 'jpeg'
        'model_id': 'my_bmshj2018_factorized',
        'algorithm': 'cw',
        'pgd': {
            'eta': 0.9
        },
        'cw': {
            'c': 1.0
        }
    }
    x_src, x_adv, x_tar, _ = direct_attack(config)