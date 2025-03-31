import warnings
warnings.filterwarnings("ignore")

import json
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

from bitstream_attack.attack import MGD, MGD2, PGD, CW
from datasets import KodakDataset
from compressor import NeuralCompressor, JpegCompressor

# Disable TF32 Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")

def load_configs(json_path):
    with open(json_path, 'r') as f:
        configs = json.load(f)
    return configs

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
        dataset = datasets.CelebA(root='../data', split='train', transform=transform, download=True)
    elif config['dataset'] == 'imagenette':
        dataset = datasets.Imagenette(root='../data', split='train', transform=transform)
    elif config['dataset'] == 'kodak':
        dataset = KodakDataset( root = '../data/kodak', transform=transform)
    else:
        raise ValueError("Invalid dataset. Use 'celeba', 'imagenette' or 'kodak'.")
    return dataset

def get_attack_algo(config, compressor):
    if config['algorithm'] == 'mgd':
        attack = MGD(model=compressor, config=config, device=device)
    elif config['algorithm'] == 'mgd2':
        attack = MGD2(model=compressor, config=config, device=device)
    elif config['algorithm'] == 'pgd':
        attack = PGD(model=compressor, config=config, device=device)
    elif config['algorithm'] == 'cw':
        attack = CW(model=compressor, config=config, device=device)
    else:
        raise ValueError("Invalid algorithm. Use 'mgd', 'pgd' or 'cw'.")
    return attack

def setup_wandb(config):
    if wandb_available:
        wandb.init(project="neural-image-compression-attack")
        wandb.config.update(config, allow_val_change=True)

def direct_attack(config):
    compressor = get_compressor(config)
    dataset = get_dataset(config)
    batch_size = 24 if config['dataset'] == 'kodak' else config['batch_size']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    attack = get_attack_algo(config, compressor)

    setup_wandb(config)

    x = dataset[0][0].unsqueeze(0).to(device)
    x_target, x_adv, x_src, loss_tracker =  attack.attack(x, dataloader, compressor, device)

    if wandb_available:
        wandb.finish()

    return x_target, x_adv, x_src, loss_tracker

def run_experiments(configs):
    for i, config in enumerate(configs):
        print(f"Running experiment {i + 1}/{len(configs)}")
        try:
            x_target, x_adv, x_src, _ = direct_attack(config)
            print(f"Experiment {i + 1} completed successfully.\n\n")
        except Exception as e:
            print(f"Experiment {i + 1} failed with error: {e}\n\n")

if __name__ == '__main__':
    # Load the run config
    config_file = 'configs.json'
    experiment_configs = load_configs(config_file)

    # Run all experiments
    run_experiments(experiment_configs)