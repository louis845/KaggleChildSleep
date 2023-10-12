import torch
import numpy as np
from model_unet import *

class MultiHeadAttn1D(torch.nn.Module):
    # assumes the input is (N, C, T)
    def __init__(self, in_channels, out_channels, key_query_channels, heads=8):
        super(MultiHeadAttn1D, self).__init__()
        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        assert key_query_channels % heads == 0, "key_query_channels must be divisible by heads"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_query_channels = key_query_channels
        self.heads = heads

        self.proj_q = torch.nn.Conv1d(in_channels, key_query_channels, kernel_size=1, bias=False)
        self.proj_k = torch.nn.Conv1d(in_channels, key_query_channels, kernel_size=1, bias=False)
        self.proj_v = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        N, C, T = x.shape

        query = self.proj_q(x)
        key = self.proj_k(x)
        value = self.proj_v(x)

        query = query.reshape(N, self.heads, self.key_query_channels // self.heads, 1, T)
        key = key.reshape(N, self.heads, self.key_query_channels // self.heads, T, 1)
        value = value.reshape(N, self.heads, self.out_channels // self.heads, T, 1)

        scaled_dot = torch.sum(query * key, dim=2, keepdim=True) / np.sqrt(self.key_query_channels // self.heads)
        attn = torch.softmax(scaled_dot, dim=3)

        out = torch.sum(attn * value, dim=3)
        assert out.shape == (N, self.heads, self.out_channels // self.heads, T)
        out = out.reshape(N, self.out_channels, T)

        return out

class AttentionBlock1DWithPositionalEncoding(torch.nn.Module):
    def __init__(self, channels, hidden_channels, key_query_channels, heads=8, dropout=0.0):
        super(AttentionBlock1DWithPositionalEncoding, self).__init__()
        self.multihead_attn = MultiHeadAttn1D(channels, hidden_channels, key_query_channels, heads)
        self.layer_norm1 = torch.nn.GroupNorm(num_channels=hidden_channels, num_groups=1)
        self.nonlin1 = torch.nn.GELU()

        self.linear = torch.nn.Conv1d(hidden_channels, channels, kernel_size=1, bias=False)
        self.layer_norm2 = torch.nn.GroupNorm(num_channels=channels, num_groups=1)
        self.nonlin2 = torch.nn.GELU()

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x_init = x
        x = self.multihead_attn(x)
        x = self.layer_norm1(x)
        x = self.nonlin1(x)

        x = self.linear(x)
        x = self.layer_norm2(x)
        x = self.dropout(x)

        return self.nonlin2(x_init + x)

class UnetHead3f(torch.nn.Module):
    def __init__(self, upconv_layers, stem_final_layer_channels,
                 kernel_size, dropout: float):
        super(UnetHead3f, self).__init__()
        assert stem_final_layer_channels % 4 == 0, "stem_final_layer_channels must be divisible by 4"

        self.upconv_layers = upconv_layers

        self.attn_stem_cat = torch.nn.Conv1d(stem_final_layer_channels * 2, stem_final_layer_channels,
                                                kernel_size=kernel_size, groups=stem_final_layer_channels // 4,
                                                bias=False, padding="same", padding_mode="replicate")

        self.upsample_conv = torch.nn.ModuleList()
        self.upsample_norms = torch.nn.ModuleList()

        for k in range(upconv_layers):
            deep_num_channels = stem_final_layer_channels // (2 ** k)
            if deep_num_channels % 4 == 0:
                num_groups = deep_num_channels // 4
            else:
                num_groups = 1
            self.upsample_conv.append(torch.nn.Conv1d(deep_num_channels, stem_final_layer_channels,
                                                      kernel_size=kernel_size, groups=num_groups, bias=False,
                                                      padding="same", padding_mode="replicate"))
            self.upsample_norms.append(torch.nn.GroupNorm(num_channels=stem_final_layer_channels, num_groups=1))

        self.cat_conv = torch.nn.ModuleList()
        self.cat_norms = torch.nn.ModuleList()
        for k in range(upconv_layers):
            self.cat_conv.append(torch.nn.Conv1d(stem_final_layer_channels * 2, stem_final_layer_channels,
                                                 kernel_size=1, bias=False, padding="same", padding_mode="replicate"))
            self.cat_norms.append(torch.nn.GroupNorm(num_channels=stem_final_layer_channels, num_groups=1))

        self.nonlin = torch.nn.GELU()
        if dropout > 0.0:
            self.dropout = torch.nn.Dropout1d(dropout)

        self.use_dropout = dropout > 0.0

    def forward(self, ret, attn_out):
        x = attn_out
        for i in range(self.upconv_layers):
            up_channels = self.upsample_conv(ret[-(i + 1)])
            up_channels = self.upsample_norms[i](up_channels)
            up_channels = self.nonlin(up_channels)
            if self.use_dropout:
                up_channels = self.dropout(up_channels)

            if i > 0:
                x = self.upsample(x)
            assert x.shape == up_channels.shape, "x.shape: {}, up_channels.shape: {}".format(x.shape, up_channels.shape)
            x = torch.cat([x, up_channels], dim=1)
            x = self.cat_conv[i](x)
            x = self.cat_norms[i](x)
            x = self.nonlin(x)
            if self.use_dropout:
                x = self.dropout(x)
        return x

    def upsample(self, x):
        return torch.nn.functional.interpolate(x, size=(x.shape[-1] * 2,), mode="linear")

    def upsample_3f(self, x):
        return torch.nn.functional.interpolate(x, size=(x.shape[-1] * 3,), mode="linear")

class Unet3fDeepSupervision(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, blocks=[4, 6, 8, 8, 8, 8],
                 bottleneck_factor=4, squeeze_excitation=True, squeeze_excitation_bottleneck_factor=4,
                 dropout=0.0, out_channels=1, use_batch_norm=False,

                 attention_channels=128, attention_heads=4,
                 attention_key_query_channels=64, attention_blocks=4):
        super(Unet3fDeepSupervision, self).__init__()
        assert kernel_size % 2 == 1, "kernel size must be odd"
        assert len(blocks) >= 6, "blocks must have at least 6 elements"

        self.input_length_multiple = 3 * (2 ** (len(blocks) - 2))

        # modules for stem and deep supervision outputs
        self.stem = ResNetBackbone(in_channels, hidden_channels, kernel_size, blocks, bottleneck_factor,
                                       squeeze_excitation, squeeze_excitation_bottleneck_factor, dropout=dropout,
                                       use_batch_norm=use_batch_norm, downsample_3f=True)
        self.pyramid_height = len(blocks)

        self.small_head = UnetHead(self.pyramid_height - 2, hidden_channels, kernel_size, use_batch_norm, dropout, initial_downsample_3f=True)
        self.mid_head = UnetHead(self.pyramid_height - 1, hidden_channels, kernel_size, use_batch_norm, dropout, initial_downsample_3f=True)
        self.large_head = UnetHead(self.pyramid_height, hidden_channels, kernel_size, use_batch_norm, dropout, initial_downsample_3f=True)

        self.final_conv_small = torch.nn.Conv1d(hidden_channels, out_channels, kernel_size=kernel_size, bias=False, padding="same", padding_mode="replicate")
        self.final_conv_mid = torch.nn.Conv1d(hidden_channels, out_channels, kernel_size=kernel_size, bias=False, padding="same", padding_mode="replicate")
        self.final_conv_large = torch.nn.Conv1d(hidden_channels, out_channels, kernel_size=kernel_size, bias=False, padding="same", padding_mode="replicate")

        self.use_dropout = dropout > 0.0

        # modules for attention and outconv
        stem_final_layer_channels = hidden_channels * (2 ** (len(blocks) - 1))
        self.attention_blocks = torch.nn.ModuleList()
        for i in range(attention_blocks):
            self.attention_blocks.append(
                AttentionBlock1DWithPositionalEncoding(channels=stem_final_layer_channels,
                                                       hidden_channels=attention_channels,
                                                       key_query_channels=attention_key_query_channels,
                                                       heads=attention_heads,
                                                       dropout=dropout)
            )
        self.no_contraction_head = UnetHead3f(self.pyramid_height - 3, stem_final_layer_channels,
                                                kernel_size, dropout)
        self.outconv = torch.nn.Conv1d(stem_final_layer_channels,
                                       out_channels, kernel_size=1,
                                       bias=True, padding="same", padding_mode="replicate")

    def forward(self, x, deep_supervision=False):
        N, C, T = x.shape
        # generate list of downsampling methods
        downsampling_methods = [0] * self.pyramid_height
        assert T % self.input_length_multiple == 0, "T must be divisible by {}".format(self.input_length_multiple)

        # run stem
        ret = self.stem(x, downsampling_methods)

        if deep_supervision:
            # upsampling path for deep supervision
            x_small = self.small_head(ret[:-2], downsampling_methods)
            x_mid = self.mid_head(ret[:-1], downsampling_methods)
            x_large = self.large_head(ret, downsampling_methods)

            # final conv
            x_small = self.final_conv_small(x_small)
            x_mid = self.final_conv_mid(x_mid)
            x_large = self.final_conv_large(x_large)
            return x_small, x_mid, x_large
        else:
            x = ret[-1] # final layer from conv stem
            for i in range(len(self.attention_blocks)):
                x = self.attention_blocks[i](x)
            x = self.no_contraction_head(ret, x)
            x = self.outconv(x)
            return x
