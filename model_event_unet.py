import torch
from model_unet import *
from model_attention_unet import *

class EventRegressorUnet(torch.nn.Module):
    def __init__(self, in_channels=1, hidden_channels=[4, 4, 8, 16, 32], kernel_size=3, blocks=[2, 2, 2, 2, 3],
                 bottleneck_factor=4, squeeze_excitation=False, squeeze_excitation_bottleneck_factor=4,
                 dropout=0.05, out_channels=2, use_batch_norm=True,

                 upconv_kernel_size=5, upconv_channels_override=None):
        super(EventRegressorUnet, self).__init__()
        assert kernel_size % 2 == 1, "kernel size must be odd"
        assert isinstance(hidden_channels, list), "hidden_channels must be a list"
        assert len(hidden_channels) == len(blocks), "hidden_channels must have the same length as blocks"
        for channel in hidden_channels:
            assert isinstance(channel, int), "hidden_channels must be a list of ints"

        self.input_length_multiple = 3 * (2 ** (len(blocks) - 2))

        # modules for stem and deep supervision outputs
        self.backbone = ResNetBackbone(in_channels, hidden_channels, kernel_size, blocks, bottleneck_factor,
                                       squeeze_excitation, squeeze_excitation_bottleneck_factor, dropout=dropout,
                                       use_batch_norm=use_batch_norm, downsample_3f=True)
        self.pyramid_height = len(blocks)
        self.upconv_head = UnetHead3f(self.pyramid_height - 1, hidden_channels,
                                     kernel_size=upconv_kernel_size, dropout=dropout, input_attn=False,
                                     target_channels_override=upconv_channels_override)

        if upconv_channels_override is not None:
            self.final_conv = torch.nn.Conv1d(upconv_channels_override, out_channels, kernel_size=1, bias=False, padding="same", padding_mode="replicate")
        else:
            self.final_conv = torch.nn.Conv1d(hidden_channels[-1], out_channels, kernel_size=1, bias=False, padding="same", padding_mode="replicate")

        self.use_dropout = dropout > 0.0

    def forward(self, x, ret_type="all"):
        assert ret_type in ["deep"], "ret_type must be one of ['deep']"

        N, C, T = x.shape
        # generate list of downsampling methods
        downsampling_methods = [0] * self.pyramid_height
        assert T % self.input_length_multiple == 0, "T must be divisible by {}".format(self.input_length_multiple)

        ret = self.backbone(x, downsampling_methods)
        x = self.upconv_head(ret)
        x = self.final_conv(x)
        return x
