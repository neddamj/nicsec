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
from datasets import KodakDataset
from compressor import NeuralCompressor, JpegCompressor

# Disable TF32 Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")

def get_compressor(config):
    if config['compressor_type'] == 'neural':
        return NeuralCompressor(model_id=config['model_id'], quality_factor=config['quality_factor'], device=device)
    elif config['compressor_type'] == 'jpeg':
        return JpegCompressor(differentiable=True, quality_factor=config['quality_factor'], image_size=config['image_size'], device=device)
    else:
        raise ValueError("Invalid compressor type. Use 'neural' or 'jpeg'.")
    
def get_dataset(config):
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])), 
        transforms.ToTensor()
        ])
    
    if config['dataset'] == 'celeba':
        dataset = datasets.CelebA(root='./data', split='train', transform=transform, download=True)
    elif config['dataset'] == 'imagenette':
        dataset = datasets.Imagenette(root='./data', split='train', transform=transform)
    elif config['dataset'] == 'kodak':
        dataset = KodakDataset( root = 'data/kodak', transform=transform)
    else:
        raise ValueError("Invalid dataset. Use 'celeba' or 'imagenette'.")
    return dataset

def get_attack_algo(config, compressor):
    if config['algorithm'] == 'mgd':
        attack = MGD(model=compressor, config=config, device=device)
    elif config['algorithm'] == 'pgd':
        attack = PGD(model=compressor, config=config, device=device)
    elif config['algorithm'] == 'cw':
        attack = CW(model=compressor, config=config, device=device)
    else:
        raise ValueError("Invalid algorithm. Use 'mgd' or 'pgd'.")
    return attack

def setup_wandb(config):
    if wandb_available:
        wandb.init(project="neural-image-compression-attack")
        wandb.config.update(config)

def direct_attack(config):
    compressor = get_compressor(config)
    dataset = get_dataset(config)
    batch_size = 24 if config['dataset'] == 'kodak' else config['batch_size']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    attack = get_attack_algo(config, compressor)

    setup_wandb(config)

    x = dataset[0][0].unsqueeze(0).to(device)
    x_src, x_adv, x_target, loss_tracker =  attack.attack(x, dataloader, compressor, device)
    return x_src, x_adv, x_target, loss_tracker

if __name__ == '__main__':
    # Run the attack
    config = {
        'lr': 3e-2,
        'batch_size': 32,
        'num_batches': 10,
        'num_steps': 5000,
        'image_size': 256, 
        'quality_factor': 1,
        'mask_type': 'dot',
        'dataset': 'kodak',        # 'celeba', imagenette' or 'kodak'
        'compressor_type': 'neural',    # 'neural' or 'jpeg'
        'model_id': 'my_bmshj2018_factorized',
        'algorithm': 'mgd',
        'pgd': {'eta': 0.1},
        'cw': {'c': 1.0}
    }
    x_src, x_adv, x_tar, _ = direct_attack(config)