import torch 
from typing import List, Dict

from diffjpeg.diffjpeg import DiffJPEG
from load_compression_models import (
    my_bmshj2018_factorized, 
    my_bmshj2018_factorized_relu,
    my_bmshj2018_hyperprior,
    my_cheng2020_anchor,
    my_cheng2020_attn,
    my_mbt2018,
    my_mbt2018_mean
)

class NeuralCompressor:
    def __init__(
            self, 
            model_id: str, 
            quality_factor: int = 2, 
            device:str = 'cpu'
            ) -> None:
        models = {
            'my_bmshj2018_factorized_relu': my_bmshj2018_factorized_relu(quality=quality_factor, pretrained=True),
            'my_bmshj2018_factorized'     : my_bmshj2018_factorized(quality=quality_factor, pretrained=True),
            'my_bmshj2018_hyperprior'     : my_bmshj2018_hyperprior(quality=quality_factor, pretrained=True),
            'my_cheng2020_anchor'         : my_cheng2020_anchor(quality=quality_factor, pretrained=True),
            'my_cheng2020_attn'           : my_cheng2020_attn(quality=quality_factor, pretrained=True),
            'my_mbt2018'                  : my_mbt2018(quality=quality_factor, pretrained=True).train(),
            'my_mbt2018_mean'             : my_mbt2018_mean(quality=quality_factor, pretrained=True)
        }
        self.model = models[model_id].train().to(torch.float32).to(device)

    def __call__(self, x: torch.Tensor):
        return self.model(x)['y_hat']

    def compress(self, x: torch.Tensor) -> Dict:
        return self.model.compress(x)

    def decompress(self, x_hat: List, shape: List) -> torch.Tensor:
        return self.model.decompress(x_hat, shape)

class JpegCompressor:
    def __init__(
            self, 
            differentiable: bool = True, 
            quality_factor: int = 80
            ) -> None:
        self.jpeg = DiffJPEG(height=256, width=256, differentiable=differentiable, quality=80)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.jpeg(x)

    def compress(self, x: torch.Tensor) -> Dict:
        return self.jpeg.compress(x)

    def decompress(self, x_hat: List, shape: List) -> torch.Tensor:
        return self.jpeg.decompress(x_hat, shape)