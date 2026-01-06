import numpy as np
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import KodakDataset
from compressor import NeuralCompressor, JpegCompressor

# Disable TF32 Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
#device = "cuda:1" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = 'cpu'

# all the parameters
my_configs = {
    "lr": 0.001,
    "batch_size": 1,
    "num_batches": 1,
    "num_steps": 20000,
    #"model_id": "my_bmshj2018_factorized_relu",
    # "model_id": "my_bmshj2018_factorized",
    # "model_id": "my_bmshj2018_hyperprior",
    "model_id": "my_mbt2018_mean",
    # "model_id": "my_mbt2018",
    # "model_id": "my_cheng2020_anchor",
    # "model_id": "my_cheng2020_attn",
    "quality_factor": 1,
    "compressor_type": "neural",
    "image_size": 256,
    "dataset": "celeba",
    "mask_type": "dot",
    "algorithm": "mgd",
    "mgd": {
        "vertical_skip": 4,
        "horizontal_skip": 4
    },
    "pgd": {"eta": 0.1},
    "cw": {"c": 1.0}
}

# initialization: NIC model (compressor), image dataset,
my_compressor = None
if my_configs['compressor_type'] == 'neural':
    my_compressor = NeuralCompressor(model_id=my_configs['model_id'], quality_factor=my_configs['quality_factor'],
                                     device=device)
elif my_configs['compressor_type'] == 'jpeg':
    my_compressor = JpegCompressor(differentiable=True, quality_factor=my_configs['quality_factor'],
                                   image_size=my_configs['image_size'], device=device)

my_image_transformation = transforms.Compose([
    transforms.Resize((my_configs['image_size'], my_configs['image_size'])), transforms.ToTensor()])

my_dataset = None
if my_configs['dataset'] == 'celeba':
    my_dataset = datasets.CelebA(root='../../data', split='train', transform=my_image_transformation, download=True)
elif my_configs['dataset'] == 'imagenette':
    my_dataset = datasets.Imagenette(root='../../data', split='train', transform=my_image_transformation)
elif my_configs['dataset'] == 'kodak':
    my_dataset = KodakDataset(root='../data/kodak', transform=my_image_transformation)
    my_configs['batch_size'] = 24
my_dataloader = DataLoader(my_dataset, batch_size=my_configs['batch_size'], shuffle=True)

grad_mask = np.ones((1, 1, my_configs['image_size'], my_configs['image_size']), dtype=np.float32)
grad_mask[:, :, 0:my_configs['image_size']:my_configs['mgd']['vertical_skip'], 0:my_configs['image_size']:my_configs['mgd']['horizontal_skip']] = 0.0
# grad_mask[1, 1, :10, :10] = 0.0
grad_mask = torch.from_numpy(grad_mask).to(device)

# attack: use the first batch of images of as targets. use other batches as source (adv starts from source).
#        the attack objective is to change B adv images to have the same bitstreams as B target images, one-to-one.
# Currently, the code exits when the first collision image pair is found, not really looking for all the B collisions.
for batch_idx, (x_src, _) in enumerate(my_dataloader):
    if batch_idx == 0:  # pick target images from the first batch. Start attacking from 2nd batch.
        x_tgt = x_src.clone().to(device)  # (B, 3, 256, 256)
        with torch.no_grad():
            bytes_tgt_list = my_compressor.compress(x_tgt)['strings'][0]   # get target bytes, list of B strings
            temp = my_compressor.compress_till_rounding(x_tgt)        # get pre-rounding emb outputs, list of L items, each item shape [B, *, ...]
            emb_tgt, emb_tgt_median = temp['y_hat'], temp['median']   # median is for debugging only, not really used.
            emb_tgt_rounded = [torch.round(item) for item in emb_tgt]
            num_emb = 0   # total number of elements in all the pre-rounding emb outputs
            for item in emb_tgt_rounded: num_emb += item[0].numel()
        continue

    B, L = x_src.shape[0], len(emb_tgt)  # number of images in a batch, number of emb items in the list

    # Get compressed bits of the src images. Embedding of source images is not used right now.
    x_src = x_src.to(device)   # source images (B, 3, 256, 256)
    with torch.no_grad():
        bytes_src_list = my_compressor.compress(x_src)['strings'][0]
        # temp = my_compressor.compress_till_rounding(x_src)
        # emb_src, emb_src_median = temp['y_hat'], temp['median']

    print(
        f'batch {batch_idx}: model={my_configs["model_id"]}, qf={my_configs["quality_factor"]}, lr={my_configs["lr"]}, '
        f'gpu={device}, date={datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # Initialize the adversarial images with the source images, to be updated in the loop
    x_adv = x_src.clone()
    x_adv.requires_grad = True

    # Setup the optimizer and LR scheduler
    optimizer = torch.optim.Adam([x_adv], lr=my_configs['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=my_configs['num_steps'] / 10)
    start_time = datetime.now()

    for iter in range(my_configs['num_steps']):
        temp = my_compressor.compress_till_rounding(x_adv)
        emb_adv, emb_adv_median = temp['y_hat'], temp['median']

        # compare emb of adv and target images to see if there are valid adv images (same bitstream as target image)
        with torch.no_grad():
            # min_diff: max float32 difference in each image (<0.5 means rounded to the same integer), the smallest one among a batch of adv images
            temp = torch.zeros((B,), dtype=torch.float32).to(device)
            for i in range(L):
                temp = torch.maximum(temp, torch.amax(torch.abs(emb_adv[i] - emb_tgt_rounded[i]), dim=tuple(range(1, emb_adv[i].dim()))))
            min_diff = torch.min(temp)
            # min_diff_rd: largest rounded integer difference for each image (0 means collision), minimum one within the batch
            temp = torch.zeros((B,), dtype=torch.float32).to(device)
            for i in range(L):
                temp = torch.maximum(temp, torch.amax(torch.abs(torch.round(emb_adv[i]) - emb_tgt_rounded[i]), dim=tuple(range(1, emb_adv[i].dim()))))
            min_diff_rd = torch.min(temp)
            min_diff_idx = torch.argmin(temp)
            # idx_identical: index of adv images with rounded emb the same as tgt emb, may collide with target, but not used now
            temp = torch.ones((B,), dtype=torch.bool).to(device)
            for i in range(L):
                temp = temp & ((torch.round(emb_adv[i])==emb_tgt_rounded[i]).all(dim=tuple(range(1, emb_adv[i].dim()))))
            idx_identical0 = torch.where(temp)[0]

            # when adv emb is close enough to tgt emb, calculate hammding distance between adv & tgt images
            hamm, hamm0_list = 10000, []
            if min_diff_rd <= 2:
                bytes_adv_list = my_compressor.compress(x_adv)['strings'][0]
                hamm = []
                for b0, b1 in zip(bytes_tgt_list, bytes_adv_list):
                    # Only compare if lengths match.
                    if len(b0) != len(b1):
                        hamm.append(len(b0) * 8)
                        continue
                    hamming_dist = 0
                    for src_byte, adv_byte in zip(b0, b1):
                        # XOR the bytes and count the number of set bits.
                        hamming_dist += bin(src_byte ^ adv_byte).count('1')
                    hamm.append(hamming_dist)
                hamm = np.array(hamm)
                hamm0_list = np.where(hamm == 0)[0]  # list of colliding adv image indices
                if len(hamm0_list) > 0: idx_identical = hamm0_list

        # if len(idx_identical) > 0:  # possibly collision image if all round(emb_adv) elements equal round(tgt)
        if len(hamm0_list) > 0:
            break

        # if no collision adv image found, then update it.
        # this kind of loss writing guarantees that each adv image's gradient is calculated with its own loss, not loss
        # of other images in this batch
        loss_tgt = torch.zeros((B,), dtype=torch.float32).to(device)
        for i in range(L):
            loss_tgt = loss_tgt + torch.sum(torch.square(emb_adv[i] - emb_tgt_rounded[i]), dim=tuple(range(1, emb_adv[i].dim())))
        loss = torch.sum(loss_tgt) / num_emb

        optimizer.zero_grad()
        x_adv.grad, = torch.autograd.grad(loss, [x_adv])
        x_adv.grad *= grad_mask
        optimizer.step()
        # scheduler.step()
        x_adv.data = torch.clamp(x_adv.data, 0, 1)

        print(f'iter {iter}: Loss {loss.item():.6f}. Emb minmax|A-T|={min_diff:.4f} rd={int(min_diff_rd)} b={min_diff_idx.item()}. '
              f'Grad: {torch.max(torch.abs(x_adv.grad)).item():.6f}. '
              f'ImgL2: |A-S|={torch.min(torch.mean(torch.square(x_adv - x_src), dim=(1,2,3))).item():.4f}, '
              f'|A-T|={torch.min(torch.mean(torch.square(x_adv - x_tgt), dim=(1,2,3))).item():.4f}. '
              f't={(datetime.now()-start_time).total_seconds()/60:.2f}')

    # out of the inner-loop due to either finding a collision or reaching the max iteration
    print(f'Final check: bitstream min Hamm(A,T)={np.min(hamm)}. Emb minmax|A-T|={min_diff:.4f} rd={int(min_diff_rd)}. '
          f'L2: |A-S|={torch.min(torch.mean(torch.square(x_adv - x_src), dim=(1,2,3))).item():.4f}, '
          f'|A-T|={torch.min(torch.mean(torch.square(x_adv - x_tgt), dim=(1,2,3))).item():.4f}')

    if len(hamm0_list) == 0:
        print('exit: no colliding adv images left, all adv images have different bitstream than the target image')
    else:
        # show and save the best colliding adv image: largest distance from tgt image
        l2dist = torch.mean(torch.square(x_adv[hamm0_list]-x_tgt[hamm0_list]), dim=(1,2,3))
        k = torch.argmax(l2dist)
        print(f'find {len(hamm0_list)} adv images with bitstream collision, best adv-tgt l2dist={l2dist[k]:.4f}')

        plt.subplot(1, 3, 1)
        plt.imshow(x_src[hamm0_list[k]].permute(1, 2, 0).cpu())
        plt.axis('off')
        plt.title('src:'+bytes_src_list[hamm0_list[k]].hex()[:10]+'...')

        plt.subplot(1, 3, 2)
        plt.imshow(x_adv[hamm0_list[k]].permute(1, 2, 0).detach().cpu())
        plt.axis('off')
        plt.title('adv:'+bytes_adv_list[hamm0_list[k]].hex()[:10]+'...')

        plt.subplot(1, 3, 3)
        plt.imshow(x_tgt[hamm0_list[k]].permute(1, 2, 0).cpu())
        plt.axis('off')
        plt.title('tgt:'+bytes_tgt_list[hamm0_list[k]].hex()[:10]+'...')

        output_dir = Path(__file__).resolve().parent.parent / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / f'collision_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches='tight')
        print(f'Saved subplot to {fig_path}')
        plt.show()
        plt.close()

    if len(hamm0_list) > 0: break  # stop if a valid collision image is found
    if batch_idx >= my_configs['num_batches'] - 1: break
