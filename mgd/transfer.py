import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from pytorch_msssim import ssim, ms_ssim
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from bitstream_attack.attack import Attack, MetricsEvaluator
from compressor import NeuralCompressor, JpegCompressor

# Disable TF32 Tensor Cores for better reproducibility
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Check if wandb is available
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb = None
    wandb_available = False

# Set device based on availability
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

def get_compressor(config):
    if config['compressor_type'] == 'neural':
        return NeuralCompressor(model_id=config['model_id'], device=device)
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
        return datasets.CelebA(root='./data', split='train', transform=transform, download=True)
    elif config['dataset'] == 'imagenette':
        return datasets.Imagenette(root='./data', split='train', transform=transform)
    else:
        raise ValueError("Invalid dataset. Use 'celeba' or 'imagenette'.")

def setup_wandb(config):
    if wandb_available:
        wandb.init(project="neural-image-compression-attack")
        wandb.config.update(config)

def direct_attack(config):
    compressor = get_compressor(config)
    dataset = get_dataset(config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    x = dataset[0][0].unsqueeze(0).to(device)
    attack = Attack(model=compressor, batch_size=config['batch_size'], device=device)
    x_src, x_adv, loss_tracker = attack.attack(x, dataloader, compressor, device, config)
    return x_src, x_adv, loss_tracker

def transfer_attack(config, attack_model, transfer_model, num_trials=2):
    success_rates = []
    for _ in range(num_trials):
        # Run direct attack with specified compressor model
        x_src, x_adv, _ = direct_attack(config)

        # Reinitialize compressor with a new model for transferability testing
        config['model_id'] = transfer_model
        transfer_compressor = get_compressor(config)
        
        # Evaluate attack success
        evaluator = MetricsEvaluator(config['batch_size'], wandb_available)
        success_rate = evaluator._calculate_success_rate(x_src, x_adv, transfer_compressor)
        success_rates.append(success_rate)
        print(f"Transfer attack successful {success_rate}/{config['batch_size']} times")
        config['model_id'] = attack_model

    # Calculate and display average ASR
    asr = sum(success_rates) / (num_trials * config['batch_size'])
    print(f'Average Attack Success Rate (ASR): {asr * 100:.2f}%')
    return asr

if __name__ == '__main__':
    # Configuration and setup for the attack
    config = {
        'lr': 3e-2,
        'batch_size': 32,
        'num_batches': 1,
        'num_steps': 20000,
        'image_size': 128,
        'quality_factor': 1,
        'mask_type': 'dot',
        'dataset': 'imagenette',        # 'celeba' or 'imagenette'
        'compressor_type': 'neural',    # 'neural' or 'jpeg'
        'scheduler_type': 'cosine',     # 'lambda' or 'cosine'
        'model_id': 'my_bmshj2018_hyperprior',
        'transfer_model_id': 'my_bmshj2018_hyperprior'
    }
    setup_wandb(config)

    # Calculate ASR over multiple trials
    attack_model = config['model_id']
    transfer_model = config['transfer_model_id']
    transfer_attack(config, attack_model, transfer_model)