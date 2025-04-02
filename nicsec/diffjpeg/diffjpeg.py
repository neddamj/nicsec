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
        self.g_a = compress_jpeg(rounding=rounding, factor=factor)
        self.g_s = decompress_jpeg(height, width, rounding=rounding,
                                          factor=factor)

    def forward(self, x):
        y_hat, shape = self.g_a(x)
        x_hat = self.g_s(y_hat, shape)
        return {
            'x_hat': x_hat,
            'y_hat': y_hat
        }
    
    def compress(self, x):
        compressed_images = []
        shape = None
        for i in range(x.shape[0]):
            single_image = x[i:i+1] 
            y_hat, shape = self.g_a(single_image)
            np_data = y_hat.cpu().detach().numpy()
            buffer = BytesIO()
            np.save(buffer, np_data)  # Save numpy array to buffer
            byte_data = buffer.getvalue()
            compressed_string = zlib.compress(byte_data)
            # Append each image's compressed data
            compressed_images.append([compressed_string])
        # Return the compressed data and single shape
        return {'strings': compressed_images, 'shape': shape}
    
    def decompress(self, strings, shape):
        decompressed_images = []
        for i, compressed_string in enumerate(strings):
            decompressed_data = zlib.decompress(compressed_string[0])
            buffer = BytesIO(decompressed_data)
            np_data = np.load(buffer)  
            tensor = torch.from_numpy(np_data)            
            # Decode the tensor back to an image
            x_hat = self.g_s(tensor, shape)
            decompressed_images.append(x_hat)
        # Stack all decompressed images along the batch dimension
        x_hat_batch = torch.cat(decompressed_images, dim=0)
        return {'x_hat': x_hat_batch}
    
    def analysis(self, x):
        y_hat, shape = self.g_a(x)
        return y_hat, shape
    
    def synthesis(self, y_hat, shape=None):
        assert shape is not None, "Shape must be provided for synthesis for JPEG compressor."
        x_hat = self.g_a(y_hat, shape)
        return x_hat