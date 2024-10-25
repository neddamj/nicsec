import torch
import torch.nn as nn
import zlib
import numpy as np
from io import BytesIO

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
        self.analysis = compress_jpeg(rounding=rounding, factor=factor)
        self.synthesis = decompress_jpeg(height, width, rounding=rounding,
                                          factor=factor)

    def forward(self, x):
        y_hat, shape = self.analysis(x)
        x_hat = self.synthesis(y_hat, shape)
        return {
            'x_hat': x_hat,
            'y_hat': y_hat
        }
    
    def compress(self, x):
        y_hat, shape = self.analysis(x)
        np_data = y_hat.cpu().detach().numpy()
        buffer = BytesIO()
        np.save(buffer, np_data)  # Save numpy array to buffer
        byte_data = buffer.getvalue()
        strings = zlib.compress(byte_data)
        return {'strings': strings, 'shape': shape}

    def decompress(self, strings, shape):
        decompressed_data = zlib.decompress(strings)
        buffer = BytesIO(decompressed_data)
        np_data = np.load(buffer)  
        tensor = torch.from_numpy(np_data)
        # Decode the tensor back to an image
        x_hat = self.synthesis(tensor, shape)
        return {'x_hat': x_hat}