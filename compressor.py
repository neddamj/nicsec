import torch 
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
    def __init__(self, model_name, quality_factor=2, device='cpu'):
        models = {
            'my_bmshj2018_factorized_relu': my_bmshj2018_factorized_relu(quality=quality_factor, pretrained=True).train().to(torch.float32).to(device),
            'my_bmshj2018_factorized'     : my_bmshj2018_factorized(quality=quality_factor, pretrained=True).train().to(torch.float32).to(device),
            'my_bmshj2018_hyperprior'     : my_bmshj2018_hyperprior(quality=quality_factor, pretrained=True).train().to(torch.float32).to(device),
            'my_cheng2020_anchor'         : my_cheng2020_anchor(quality=quality_factor, pretrained=True).train().to(torch.float32).to(device),
            'my_cheng2020_attn'           : my_cheng2020_attn(quality=quality_factor, pretrained=True).train().to(torch.float32).to(device),
            'my_mbt2018'                  : my_mbt2018(quality=quality_factor, pretrained=True).train().to(torch.float32).to(device),
            'my_mbt2018_mean'             : my_mbt2018_mean(quality=quality_factor, pretrained=True).to(torch.float32).train().to(device)
        }
        self.model = models[model_name]

    def __call__(self, x):
        return self.model(x)['y_hat']
    
    def run(self, x):
        return self.model(x)

    def compress(self, x):
        return self.model.compress(x)['strings']

    def decompress(self, x_hat):
        return self.model.decompress(x_hat)

class JpegCompressor:
    def __init__(self):
        pass

    def __call__(self, x):
        pass

    def compress(self):
        pass

    def decompress(self):
        pass
