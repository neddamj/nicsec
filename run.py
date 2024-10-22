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
from load_compression_models import (
    my_bmshj2018_factorized, 
    my_bmshj2018_factorized_relu,
    my_bmshj2018_hyperprior,
    my_cheng2020_anchor,
    my_cheng2020_attn,
    my_mbt2018,
    my_mbt2018_mean
)

# Disable TF32 Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda' 
elif torch.backends.mps.is_available():
    device = 'mps'

def get_model(model_id, device):
    models = {
        'my_bmshj2018_factorized_relu': my_bmshj2018_factorized_relu(quality=2, pretrained=True),
        'my_bmshj2018_factorized'     : my_bmshj2018_factorized(quality=2, pretrained=True),
        'my_bmshj2018_hyperprior'     : my_bmshj2018_hyperprior(quality=2, pretrained=True),
        'my_cheng2020_anchor'         : my_cheng2020_anchor(quality=2, pretrained=True),
        'my_cheng2020_attn'           : my_cheng2020_attn(quality=2, pretrained=True),
        'my_mbt2018'                  : my_mbt2018(quality=2, pretrained=True),
        'my_mbt2018_mean'             : my_mbt2018_mean(quality=2, pretrained=True),
    }
    return models[model_id].train().to(torch.float32).to(device)

def setup_wandb(config, wandb_available):
    # Setup wandb logging
    if wandb_available:
        wandb.init(project="neural-image-compression-attack")
        wandb.config.update(config)

def run_attack(attack, x, dataloader, num_batches, net, device, config):
    for i, (x_hat, _) in enumerate(dataloader):
        x_hat = x_hat.to(device)
        x_src = x_hat.clone()
        x_hat.requires_grad = True
        mask = attack.init_mask(x_hat, config['mask_type'])

        optimizer = torch.optim.Adam([x_hat], lr=config['lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=config['num_steps'] / 10)
        x_adv, loss_tracker = attack.run(x, x_hat, optimizer=optimizer, scheduler=scheduler, num_steps=config['num_steps'], mask=mask)

        with torch.no_grad():
            output = net(x_adv)['x_hat']

        attack.batch_eval(x, x_adv, output)
        if i == num_batches - 1:
            break
    attack.global_eval()

def main(config):
    net = get_model(config['model_id'], device)

    transform = transforms.Compose([transforms.Resize((config['image_size'], config['image_size'])), transforms.ToTensor()])
    dataset = datasets.CelebA(root='./data', split='train', transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    setup_wandb(config, wandb_available)

    x = dataset[0][0].unsqueeze(0).to(device)
    attack = Attack(model=net, batch_size=config['batch_size'], device=device)
    x_src, x_adv, loss_tracker =  attack.attack(x, dataloader, net, device, config)

if __name__ == '__main__':
    ''' TODO: run the attack over multiple datasets'''
    # Run the attack
    config = {
        'lr': 3e-2,
        'batch_size': 32,
        'num_batches': 5,
        'num_steps': 500,
        'image_size': 256,
        'mask_type': 'dot',    
        'model_id': 'my_bmshj2018_factorized_relu'
    }
    main(config)