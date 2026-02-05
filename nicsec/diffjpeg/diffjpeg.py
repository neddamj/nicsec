import torch
import torch.nn as nn
import zlib
import numpy as np

from .modules import compress_jpeg, decompress_jpeg
from .utils import diff_round, quality_to_factor

class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=True, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.g_a = compress_jpeg(rounding=rounding, factor=factor)
        self.g_s = decompress_jpeg(height, width, rounding=rounding,
                                          factor=factor)

        my_rounding = lambda x: x  # identity function
        self.my_g_a = compress_jpeg(rounding=my_rounding, factor=factor)

    def compress_till_rounding(self, x):
        y_hat, shape = self.my_g_a(x)
        return {'y_hat': [y_hat], 'median': [shape]}
    
    def forward(self, x):
        y_hat, shape = self.g_a(x)
        x_hat = self.g_s(y_hat, shape)
        return {
            'x_hat': x_hat,
            'y_hat': y_hat
        }
    
    def compress(self, x):
        batch_size = x.shape[0]
        compressed_images = []
        shape = None
        for i in range(batch_size):
            single_image = x[i:i + 1]
            y_hat, shape = self.g_a(single_image)
            np_data = np.ascontiguousarray(y_hat.detach().cpu().numpy())
            byte_data = np_data.astype(np.int32).reshape(-1).tobytes(order="C")
            compressed_string = zlib.compress(byte_data)
            compressed_images.append(compressed_string)
        return {'strings': [compressed_images], 'shape': shape}
    
    def decompress(self, strings, shape):
        if not isinstance(strings, list) or len(strings) != 1:
            raise ValueError("DiffJPEG.decompress expects strings as a list of length 1.")

        compressed_images = strings[0]
        if torch.is_tensor(shape):
            shape_list = [int(s) for s in shape.tolist()]
        else:
            shape_list = [int(s) for s in shape]

        total_blocks = sum(shape_list)
        expected_elems = total_blocks * 8 * 8
        device = next(self.g_s.parameters()).device

        decompressed_images = []
        for compressed_string in compressed_images:
            raw = zlib.decompress(compressed_string)
            # Raw int32 layout mirrors compress() byte packing (C-order).
            np_data = np.frombuffer(raw, dtype=np.int32)
            if np_data.size != expected_elems:
                raise ValueError(
                    f"Unexpected decompressed size: got {np_data.size}, expected {expected_elems}."
                )
            np_data = np_data.reshape(1, total_blocks, 8, 8)
            tensor = torch.from_numpy(np_data).to(device=device, dtype=torch.float32)
            x_hat = self.g_s(tensor, shape_list)
            decompressed_images.append(x_hat)

        x_hat_batch = torch.cat(decompressed_images, dim=0)
        return {'x_hat': x_hat_batch}
