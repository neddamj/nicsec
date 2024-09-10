from compressai.models import FactorizedPrior
import torch

from typing import Dict

class MyFactorizedPrior(FactorizedPrior):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> Dict:
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }