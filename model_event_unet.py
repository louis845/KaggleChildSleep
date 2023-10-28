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

class EventConfidenceUnet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, blocks=[4, 6, 8, 8, 8, 8],
                 bottleneck_factor=4, squeeze_excitation=True, squeeze_excitation_bottleneck_factor=4,
                 dropout=0.0, use_batch_norm=False, # settings for stem (backbone)

                 attention_channels=128, attention_heads=4, # settings for attention upsampling module
                 attention_key_query_channels=64, attention_blocks=4,
                 expected_attn_input_length=17280, dropout_pos_embeddings=False,
                 attn_out_channels=1, attention_bottleneck=None):
        super(EventConfidenceUnet, self).__init__()
        assert kernel_size % 2 == 1, "kernel size must be odd"
        assert len(blocks) >= 6, "blocks must have at least 6 elements"
        assert isinstance(hidden_channels, list), "hidden_channels must be a list"
        assert len(hidden_channels) == len(blocks), "hidden_channels must have the same length as blocks"
        for channel in hidden_channels:
            assert isinstance(channel, int), "hidden_channels must be a list of ints"

        self.input_length_multiple = 3 * (2 ** (len(blocks) - 2))

        # modules for stem and deep supervision outputs
        self.stem = ResNetBackbone(in_channels, hidden_channels, kernel_size, blocks, bottleneck_factor,
                                       squeeze_excitation, squeeze_excitation_bottleneck_factor, dropout=dropout,
                                       use_batch_norm=use_batch_norm, downsample_3f=True)
        self.pyramid_height = len(blocks)
        self.use_dropout = dropout > 0.0

        # modules for attention and outconv
        stem_final_layer_channels = hidden_channels[-1]

        if attention_bottleneck is not None:
            assert isinstance(attention_bottleneck, int), "bottleneck must be an integer"
            self.attn_bottleneck_in = torch.nn.Conv1d(stem_final_layer_channels, attention_bottleneck, kernel_size=1, bias=False)
            self.attn_bottleneck_in_norm = torch.nn.BatchNorm1d(attention_bottleneck)
            self.attn_bottleneck_in_nonlin = torch.nn.GELU()
            self.attn_bottleneck_out = torch.nn.Conv1d(attention_bottleneck, stem_final_layer_channels, kernel_size=1, bias=False)
            self.attn_bottleneck_out_norm = torch.nn.GroupNorm(num_channels=stem_final_layer_channels, num_groups=1)
            self.attn_bottleneck_out_nonlin = torch.nn.GELU()

        self.attention_blocks = torch.nn.ModuleList()
        for i in range(attention_blocks):
            self.attention_blocks.append(
                AttentionBlock1DWithPositionalEncoding(channels=stem_final_layer_channels,
                                                       hidden_channels=attention_channels,
                                                       key_query_channels=attention_key_query_channels,
                                                       heads=attention_heads,
                                                       dropout=dropout,
                                                       learned_embeddings=True,
                                                       input_length=expected_attn_input_length // (3 * (2 ** (len(blocks) - 2))),
                                                       dropout_pos_embeddings=dropout_pos_embeddings)
            )
        self.no_contraction_head = UnetHead3f(self.pyramid_height, hidden_channels,
                                                kernel_size=1, dropout=dropout)
        self.outconv = torch.nn.Conv1d(stem_final_layer_channels,
                                       attn_out_channels, kernel_size=1,
                                       bias=True, padding="same", padding_mode="replicate")
        self.expected_attn_input_length = expected_attn_input_length
        self.attention_bottleneck = attention_bottleneck

    def forward(self, x, ret_type="attn"):
        assert ret_type in ["attn"], "ret_type must be one of ['attn']"

        N, C, T = x.shape
        # generate list of downsampling methods
        downsampling_methods = [0] * self.pyramid_height
        assert T % self.input_length_multiple == 0, "T must be divisible by {}".format(self.input_length_multiple)

        if ret_type != "deep":
            assert T == self.expected_attn_input_length, "T: {}, self.expected_attn_input_length: {}".format(T, self.expected_attn_input_length)

        # run stem
        ret = self.stem(x, downsampling_methods)

        x = ret[-1]  # final layer from conv stem
        if self.attention_bottleneck is not None:
            x = self.attn_bottleneck_in(x)
            x = self.attn_bottleneck_in_norm(x)
            x = self.attn_bottleneck_in_nonlin(x)
            x = self.attn_bottleneck_out(x)
            x = self.attn_bottleneck_out_norm(x)
            x = self.attn_bottleneck_out_nonlin(x)

        for i in range(len(self.attention_blocks)):
            x = self.attention_blocks[i](x)

        x = self.no_contraction_head(ret, x)
        x = self.outconv(x)
        return x, None, None, None