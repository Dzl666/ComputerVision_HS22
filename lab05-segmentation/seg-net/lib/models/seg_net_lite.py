from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SegNetLite(nn.Module):

    def __init__(self, kernel_sizes=[3, 3, 3, 3], down_filter_sizes=[32, 64, 128, 256],
            up_filter_sizes=[128, 64, 32, 32], conv_paddings=[1, 1, 1, 1],
            pooling_kernel_sizes=[2, 2, 2, 2], pooling_strides=[2, 2, 2, 2], **kwargs):
        """Initialize SegNet Module

        Args:
            kernel_sizes: kernel sizes for each convolutional layer in downsample/upsample path.
            down_filter_sizes: number of filters (out channels) of each convolutional layer in the downsample path.
            up_filter_sizes : number of filters (out channels) of each convolutional layer in the upsample path.
            conv_paddings: paddings for each convolutional layer in downsample/upsample path.
            pooling_kernel_sizes: kernel sizes for each max-pooling layer and its max-unpooling layer.
            pooling_strides: strides for each max-pooling layer and its max-unpooling layer.
        """
        super(SegNetLite, self).__init__()
        self.num_down_layers = len(kernel_sizes)
        self.num_up_layers = len(kernel_sizes)

        input_size = 3 # initial number of input channels
        # Construct downsampling layers.
        # Blocks of the downsampling path should have the following output dimension (igoring batch dimension):
        # 3 x 64 x 64 (input) -> 32 x 32 x 32 -> 64 x 16 x 16 -> 128 x 8 x 8 -> 256 x 4 x 4
        # each block should consist of: Conv2d -> BatchNorm2d->ReLU -> MaxPool2d
        layers_conv_down = [torch.nn.Conv2d(3, down_filter_sizes[0], kernel_sizes[0], padding=conv_paddings[0])]
        for i in range(1, self.num_down_layers):
            layers_conv_down.append(torch.nn.Conv2d(
                down_filter_sizes[i-1], down_filter_sizes[i], 
                kernel_sizes[i], padding=conv_paddings[i]
            ))
        layers_bn_down = [
            torch.nn.BatchNorm2d(down_filter_sizes[i]) for i in range(self.num_down_layers)
        ]
        layers_pooling = [
            torch.nn.MaxPool2d(
                pooling_kernel_sizes[i],
                stride=pooling_strides[i],
                return_indices=True
            ) for i in range(self.num_down_layers)
        ]
        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # package can track gradients and update parameters of these layers
        self.layers_conv_down = nn.ModuleList(layers_conv_down)
        self.layers_bn_down = nn.ModuleList(layers_bn_down)
        self.layers_pooling = nn.ModuleList(layers_pooling)

        # Construct upsampling layers
        # Blocks of the upsampling path should have the following output dimension (igoring batch dimension):
        # 256 x 4 x 4 (input) -> 128 x 8 x 8 -> 64 x 16 x 16 -> 32 x 32 x 32 -> 32 x 64 x 64
        # each block should consist of: MaxUnpool2d -> Conv2d -> BatchNorm2d->ReLU
        layers_unpooling = [
            torch.nn.MaxUnpool2d(
                pooling_kernel_sizes[i],
                stride=pooling_strides[i]
            ) for i in range(self.num_up_layers)
        ]
        layers_conv_up = [
            torch.nn.Conv2d(256, up_filter_sizes[0], kernel_sizes[0], padding=conv_paddings[0])
        ]
        for i in range(1, self.num_up_layers):
            layers_conv_up.append(torch.nn.Conv2d(
                up_filter_sizes[i-1], up_filter_sizes[i],
                kernel_sizes[i], padding=conv_paddings[i]
            ))

        layers_bn_up = [
            torch.nn.BatchNorm2d(up_filter_sizes[i]) for i in range(self.num_up_layers)
        ]
        
        # Convert Python list to nn.ModuleList
        self.layers_unpooling = nn.ModuleList(layers_unpooling)
        self.layers_conv_up = nn.ModuleList(layers_conv_up)
        self.layers_bn_up = nn.ModuleList(layers_bn_up)

        self.relu = nn.ReLU(True)

        # Implement a final 1x1 convolution to to get the logits of 11 classes (background + 10 digits)
        self.final_conv = torch.nn.Conv2d(32, 11, kernel_size=1)
        self.actv = torch.nn.Softmax(dim=0)

    def forward(self, x):
        # downsample period
        idxs = []
        for i in range(self.num_down_layers):
            conv_x = self.layers_conv_down[i](x)
            bn_x = self.relu(self.layers_bn_down[i](conv_x))
            x, idx = self.layers_pooling[i](bn_x)
            idxs.append(idx)
        # upsample period
        for i in range(self.num_up_layers):
            unpool_x = self.layers_unpooling[i](x, idxs[3-i])
            conv_x = self.layers_conv_up[i](unpool_x)
            x = self.relu(self.layers_bn_up[i](conv_x))
        
        output = self.final_conv(x)
        # return self.actv(output)
        return output


def get_seg_net(**kwargs):
    model = SegNetLite(**kwargs)
    return model
