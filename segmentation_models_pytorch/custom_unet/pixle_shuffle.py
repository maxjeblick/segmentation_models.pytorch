import torch
from torch import relu
import torch.nn as nn


class PixelShuffleUpsample(nn.Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."

    def __init__(self, in_channels: int, out_channels: int, scale: int = 2, blur: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3,3),
                              stride=1,
                              padding=(1, 1))
        self.pixel_shuffle = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.blur = blur
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.avg_pool = nn.AvgPool2d(2, stride=1)

    def forward(self, x):
        x = self.conv(x)
        x = relu(x)
        x = self.pixel_shuffle(x)
        if self.blur:
            x = self.pad(x)
            x = self.avg_pool(x)
        return x

if __name__ == '__main__':
    x = torch.rand((1, 32, 64, 64))
    print(x.shape)

    upsampler = PixelShuffleUpsample(in_channels=32, out_channels=128)
    print(upsampler(x).shape)