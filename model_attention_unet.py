import torch
import numpy as np
from model_unet import *

class SymmetricLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_groups=2):
        super(SymmetricLinear, self).__init__()
        assert in_dim % num_groups == 0, "in_dim must be divisible by num_groups"
        assert out_dim % num_groups == 0, "out_dim must be divisible by num_groups"

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_groups = num_groups
        self.out_roll_stride = out_dim // num_groups

        weight_init = torch.zeros((in_dim // num_groups, out_dim), dtype=torch.float32)
        stdv = 1.0 / np.sqrt(in_dim)
        weight_init.uniform_(-stdv, stdv)
        self.weight = torch.nn.Parameter(weight_init)

    def forward(self, x):
        cp_weight = torch.cat([torch.roll(self.weight, k * self.out_roll_stride, dims=1) for k in range(self.num_groups)], dim=0)
        return torch.matmul(x, cp_weight)


class PairwiseLengthAttn1D(torch.nn.Module):
    # assumes the input is (N, C, T)
    def __init__(self, in_channels, out_channels, key_query_channels, input_length, heads=8, num_mix=1, dropout=0.0):
        super(PairwiseLengthAttn1D, self).__init__()
        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        assert key_query_channels % heads == 0, "key_query_channels must be divisible by heads"
        assert num_mix > 0, "num_mix must be > 0"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_query_channels = key_query_channels
        self.input_length = input_length
        self.heads = heads
        self.num_mix = num_mix
        self.dropout = dropout
        self.use_dropout = dropout > 0.0

        self.proj_q = torch.nn.Conv1d(in_channels, key_query_channels, kernel_size=1, bias=False)
        self.proj_k = torch.nn.Conv1d(in_channels, key_query_channels, kernel_size=1, bias=False)
        self.proj_v = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        self.register_buffer("pairwise_distance", (torch.abs(torch.arange(input_length, dtype=torch.float32) -
                                                            torch.arange(input_length, dtype=torch.float32).unsqueeze(-1)) / (input_length - 1))
                                                            .unsqueeze(0).unsqueeze(0)) # (1, 1, T, T)


        self.mix1 = torch.nn.ModuleList()
        self.mix1_norm = torch.nn.ModuleList()
        self.mix2 = torch.nn.ModuleList()
        self.mix2_norm = torch.nn.ModuleList()

        for k in range(num_mix):
            mix1_module = SymmetricLinear(input_length * 2, input_length * 2, num_groups=input_length)
            if k == num_mix - 1:
                mix2_module = SymmetricLinear(input_length * 2, input_length, num_groups=input_length)
            else:
                mix2_module = SymmetricLinear(input_length * 2, input_length * 2, num_groups=input_length)
            self.mix1.append(mix1_module)
            self.mix2.append(mix2_module)

            self.mix1_norm.append(torch.nn.LayerNorm([input_length, input_length * 2]))
            if k != num_mix - 1:
                self.mix2_norm.append(torch.nn.LayerNorm([input_length, input_length * 2]))
        self.mix_nonlin = torch.nn.GELU()

        if self.use_dropout:
            self.dropout_layer = torch.nn.Dropout(dropout)

    def forward(self, x):
        N, C, T = x.shape
        assert T == self.input_length, "T must be equal to self.input_length"

        query = self.proj_q(x)
        key = self.proj_k(x)
        value = self.proj_v(x)

        query = query.reshape(N, self.heads, self.key_query_channels // self.heads, T)
        key = key.reshape(N, self.heads, self.key_query_channels // self.heads, T)
        value = value.reshape(N, self.heads, self.out_channels // self.heads, T)

        query = query.permute(0, 1, 3, 2) # (N, H, C, T) -> (N, H, T, C)
        value = value.permute(0, 1, 3, 2) # (N, H, C, T) -> (N, H, T, C)
        scaled_dot = torch.matmul(query, key) / np.sqrt(self.key_query_channels // self.heads) # (N, H, T, C) * (N, H, C, T) -> (N, H, T, T)
        pairwise_info = torch.stack((scaled_dot, self.pairwise_distance.expand(N, self.heads, -1, -1)), dim=-1) # (N, H, T, T, 2)
        # mixer
        for k in range(self.num_mix):
            pairwise_info = pairwise_info.view(N, self.heads, T, T * 2) # (N, H, T, T, 2) -> (N, H, T, T * 2)
            pairwise_info = self.mix1[k](pairwise_info)
            pairwise_info = self.mix1_norm[k](pairwise_info)

            # swap the T
            # (N, H, T, T * 2) -> (N, H, T, T, 2) -> (N, H, T, T, 2)
            pairwise_info = pairwise_info.view(N, self.heads, T, T, 2).permute(0, 1, 3, 2, 4).contiguous()
            pairwise_info = self.mix_nonlin(pairwise_info).view(N, self.heads, T, T * 2) # (N, H, T, T, 2) -> (N, H, T, T * 2)
            if self.use_dropout:
                pairwise_info = self.dropout_layer(pairwise_info)
            pairwise_info = self.mix2[k](pairwise_info)
            if k != self.num_mix - 1:
                pairwise_info = self.mix2_norm[k](pairwise_info)

                # swap back the T
                pairwise_info = pairwise_info.view(N, self.heads, T, T, 2).permute(0, 1, 3, 2, 4).contiguous()
                pairwise_info = self.mix_nonlin(pairwise_info)
                if self.use_dropout:
                    pairwise_info = self.dropout_layer(pairwise_info)
            else:
                # swap back the T
                pairwise_info = pairwise_info.permute(0, 1, 3, 2).contiguous()

        if self.use_dropout and self.training:
            dropout_mask = torch.where(torch.rand(N, self.heads, T, T, device=x.device) < self.dropout, -np.inf, 0.0)
            attn = torch.nn.functional.softmax(pairwise_info + dropout_mask, dim=-1)  # (N, H, T, T)
        else:
            attn = torch.nn.functional.softmax(pairwise_info, dim=-1) # (N, H, T, T)
        out = torch.matmul(attn, value) # (N, H, T, T) * (N, H, T, C) -> (N, H, T, C)

        out = out.permute(0, 1, 3, 2) # (N, H, T, C) -> (N, H, C, T)
        out = out.reshape(N, self.out_channels, T) # (N, H, C, T) -> (N, C, T)
        return out

class MultiHeadAttn1D(torch.nn.Module):
    # assumes the input is (N, C, T)
    def __init__(self, in_channels, out_channels, key_query_channels, heads=8, dropout=0.0):
        super(MultiHeadAttn1D, self).__init__()
        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        assert key_query_channels % heads == 0, "key_query_channels must be divisible by heads"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_query_channels = key_query_channels
        self.heads = heads
        self.dropout = dropout

        self.proj_q = torch.nn.Conv1d(in_channels, key_query_channels, kernel_size=1, bias=False)
        self.proj_k = torch.nn.Conv1d(in_channels, key_query_channels, kernel_size=1, bias=False)
        self.proj_v = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        N, C, T = x.shape

        query = self.proj_q(x)
        key = self.proj_k(x)
        value = self.proj_v(x)

        query = query.reshape(N, self.heads, self.key_query_channels // self.heads, T)
        key = key.reshape(N, self.heads, self.key_query_channels // self.heads, T)
        value = value.reshape(N, self.heads, self.out_channels // self.heads, T)

        query = query.permute(0, 1, 3, 2) # (N, H, C, T) -> (N, H, T, C)
        value = value.permute(0, 1, 3, 2) # (N, H, C, T) -> (N, H, T, C)
        scaled_dot = torch.matmul(query, key) / np.sqrt(self.key_query_channels // self.heads) # (N, H, T, C) * (N, H, C, T) -> (N, H, T, T)
        if (self.dropout > 0.0) and self.training:
            dropout_mask = torch.where(torch.rand(N, self.heads, T, T, device=x.device) < self.dropout, -np.inf, 0.0)
            attn = torch.nn.functional.softmax(scaled_dot + dropout_mask, dim=-1) # (N, H, T, T)
        else:
            attn = torch.nn.functional.softmax(scaled_dot, dim=-1) # (N, H, T, T)
        out = torch.matmul(attn, value) # (N, H, T, T) * (N, H, T, C) -> (N, H, T, C)

        out = out.permute(0, 1, 3, 2) # (N, H, T, C) -> (N, H, C, T)
        out = out.reshape(N, self.out_channels, T) # (N, H, C, T) -> (N, C, T)
        return out

class AttentionBlock1DWithPositionalEncoding(torch.nn.Module):
    def __init__(self, channels, hidden_channels, key_query_channels,
                 heads=8, dropout=0.0, attention_mode="learned", input_length=1000,
                 dropout_pos_embeddings=False, attn_dropout=0.0):
        super(AttentionBlock1DWithPositionalEncoding, self).__init__()
        assert channels % 2 == 0, "channels must be divisible by 2"
        assert attention_mode in ["learned", "length"], "attention_mode must be one of ['learned', 'length']"
        if dropout_pos_embeddings:
            assert attention_mode == "learned", "dropout_pos_embeddings can only be True if attention_mode is 'learned'"
        self.input_length = input_length

        if attention_mode == "learned":
            self.multihead_attn = MultiHeadAttn1D(channels + (channels // 2), hidden_channels, key_query_channels, heads, dropout=attn_dropout)
        else:
            self.length_attn = PairwiseLengthAttn1D(channels, hidden_channels, key_query_channels, input_length, heads, dropout=attn_dropout)
        self.layer_norm1 = torch.nn.GroupNorm(num_channels=hidden_channels, num_groups=1)
        self.nonlin1 = torch.nn.GELU()

        self.linear = torch.nn.Conv1d(hidden_channels, channels, kernel_size=1, bias=False)
        self.layer_norm2 = torch.nn.GroupNorm(num_channels=channels, num_groups=1)
        self.nonlin2 = torch.nn.GELU()

        self.dropout = torch.nn.Dropout(dropout)

        self.attention_mode = attention_mode
        if attention_mode == "learned":
            self.positional_embedding = torch.nn.Parameter(torch.randn((1, channels // 2, input_length)))

        self.dropout_pos_embeddings = dropout_pos_embeddings

    def forward(self, x):
        assert x.shape[-1] == self.input_length, "x.shape: {}, self.input_length: {}".format(x.shape, self.input_length)
        x_init = x
        if self.attention_mode == "learned":
            positional_embedding = self.positional_embedding.expand(x.shape[0], -1, -1)

            if self.training and self.dropout_pos_embeddings:
                positional_embedding_mean = torch.mean(positional_embedding, dim=-1, keepdim=True)
                #lam = torch.rand((positional_embedding.shape[0], 1, 1), device=positional_embedding.device, dtype=torch.float32)
                lam = np.random.beta(0.3, 0.3, size=(positional_embedding.shape[0], 1, 1))
                lam = torch.tensor(lam, device=positional_embedding.device, dtype=torch.float32)
                positional_embedding = lam * positional_embedding + (1.0 - lam) * positional_embedding_mean

            x = self.multihead_attn(torch.cat([x, positional_embedding], dim=1))
        else:
            x = self.length_attn(x)

        x = self.layer_norm1(x)
        x = self.nonlin1(x)

        x = self.linear(x)
        x = self.layer_norm2(x)
        x = self.dropout(x)

        return self.nonlin2(x_init + x)

class UnetHead3f(torch.nn.Module):
    def __init__(self, upconv_layers, stem_hidden_channels,
                 kernel_size, dropout: float, input_attn=True, target_channels_override=None, use_batch_norm=True):
        super(UnetHead3f, self).__init__()
        if input_attn:
            assert upconv_layers <= len(stem_hidden_channels), "upconv_layers must be <= len(stem_hidden_channels)"
        else:
            assert upconv_layers < len(stem_hidden_channels), "upconv_layers must be < len(stem_hidden_channels)"

        stem_final_layer_channels = stem_hidden_channels[-1]
        if target_channels_override is not None:
            assert isinstance(target_channels_override, int), "target_channels_override must be an integer"
            self.initial_conv = torch.nn.Conv1d(stem_final_layer_channels, target_channels_override,
                                                kernel_size=1, bias=False, padding="same", padding_mode="replicate")
            if use_batch_norm:
                self.initial_norm = torch.nn.BatchNorm1d(target_channels_override)
            else:
                self.initial_norm = torch.nn.GroupNorm(num_channels=target_channels_override, num_groups=1)
            stem_final_layer_channels = target_channels_override
        assert stem_final_layer_channels % 4 == 0, "stem_final_layer_channels must be divisible by 4"

        self.upconv_layers = upconv_layers

        self.upsample_conv = torch.nn.ModuleList()
        self.upsample_norms = torch.nn.ModuleList()

        for k in range(upconv_layers):
            if input_attn:
                deep_num_channels = stem_hidden_channels[-(k + 1)]
            else:
                deep_num_channels = stem_hidden_channels[-(k + 2)]
            if np.gcd(deep_num_channels, stem_final_layer_channels) % 4 == 0:
                num_groups = np.gcd(deep_num_channels, stem_final_layer_channels) // 4
            else:
                num_groups = 1
            self.upsample_conv.append(torch.nn.Conv1d(deep_num_channels, stem_final_layer_channels,
                                                      kernel_size=kernel_size, groups=num_groups, bias=False,
                                                      padding="same", padding_mode="replicate")) # channel upsampling
            if use_batch_norm:
                self.upsample_norms.append(torch.nn.BatchNorm1d(stem_final_layer_channels))
            else:
                self.upsample_norms.append(torch.nn.GroupNorm(num_channels=stem_final_layer_channels, num_groups=1))

        self.cat_conv = torch.nn.ModuleList()
        self.cat_norms = torch.nn.ModuleList()
        for k in range(upconv_layers):
            self.cat_conv.append(torch.nn.Conv1d(stem_final_layer_channels * 2, stem_final_layer_channels,
                                                 kernel_size=1, bias=False, padding="same", padding_mode="replicate"))
            if use_batch_norm:
                self.cat_norms.append(torch.nn.BatchNorm1d(stem_final_layer_channels))
            else:
                self.cat_norms.append(torch.nn.GroupNorm(num_channels=stem_final_layer_channels, num_groups=1))

        self.nonlin = torch.nn.GELU()
        if dropout > 0.0:
            self.dropout = torch.nn.Dropout1d(dropout)

        self.use_dropout = dropout > 0.0
        self.input_attn = input_attn
        self.target_channels_override = target_channels_override
        self.upsamples = []

        if input_attn:
            for k in range(upconv_layers):
                if k == 0:
                    self.upsamples.append(1)
                elif k == len(stem_hidden_channels) - 1:
                    self.upsamples.append(3)
                else:
                    self.upsamples.append(2)
        else:
            for k in range(upconv_layers):
                if k == len(stem_hidden_channels) - 2:
                    self.upsamples.append(3)
                else:
                    self.upsamples.append(2)

    def forward(self, ret, attn_out=None):
        if self.input_attn:
            assert attn_out is not None, "attn_out must be given if input_attn is True"
            x = attn_out
        else:
            assert attn_out is None, "attn_out must be None if input_attn is False"
            x = ret[-1]

        if self.target_channels_override is not None: # fix the number of channels
            x = self.initial_conv(x)
            x = self.initial_norm(x)
            x = self.nonlin(x)
            if self.use_dropout:
                x = self.dropout(x)

        for i in range(self.upconv_layers):
            if self.input_attn:
                up_channels = self.upsample_conv[i](ret[-(i + 1)]) # upsample the number of channels
            else:
                up_channels = self.upsample_conv[i](ret[-(i + 2)])
            up_channels = self.upsample_norms[i](up_channels)
            up_channels = self.nonlin(up_channels)
            if self.use_dropout:
                up_channels = self.dropout(up_channels)

            if self.upsamples[i] == 2: # upsample the temporal dimension of the previous layer
                x = self.upsample(x)
            elif self.upsamples[i] == 3:
                x = self.upsample_3f(x)

            # concat and mix information
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
                 dropout=0.0, out_channels=1, use_batch_norm=False, # settings for stem (backbone)

                 deep_supervision_kernel_size=1, deep_supervision_contraction=True, deep_supervision_channels_override=None,

                 attention_channels=128, attention_heads=4, # settings for attention upsampling module
                 attention_key_query_channels=64, attention_blocks=4,
                 expected_attn_input_length=17280, dropout_pos_embeddings=False,
                 attn_out_channels=1, attention_bottleneck=None):
        super(Unet3fDeepSupervision, self).__init__()
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

        # modules for deep supervision expanding path
        if deep_supervision_contraction:
            self.small_head = UnetHead(self.pyramid_height - 2, hidden_channels[:-2], deep_supervision_kernel_size, use_batch_norm, dropout, initial_downsample_3f=True)
            self.mid_head = UnetHead(self.pyramid_height - 1, hidden_channels[:-1], deep_supervision_kernel_size, use_batch_norm, dropout, initial_downsample_3f=True)
            self.large_head = UnetHead(self.pyramid_height, hidden_channels, deep_supervision_kernel_size, use_batch_norm, dropout, initial_downsample_3f=True)

            self.final_conv_small = torch.nn.Conv1d(hidden_channels[0], out_channels, kernel_size=1, bias=False, padding="same", padding_mode="replicate")
            self.final_conv_mid = torch.nn.Conv1d(hidden_channels[0], out_channels, kernel_size=1, bias=False, padding="same", padding_mode="replicate")
            self.final_conv_large = torch.nn.Conv1d(hidden_channels[0], out_channels, kernel_size=1, bias=False, padding="same", padding_mode="replicate")
        else:
            self.small_head = UnetHead3f(self.pyramid_height - 3, hidden_channels[:-2],
                                         kernel_size=deep_supervision_kernel_size, dropout=dropout, input_attn=False,
                                         target_channels_override=deep_supervision_channels_override)
            self.mid_head = UnetHead3f(self.pyramid_height - 2, hidden_channels[:-1],
                                         kernel_size=deep_supervision_kernel_size, dropout=dropout, input_attn=False,
                                         target_channels_override=deep_supervision_channels_override)
            self.large_head = UnetHead3f(self.pyramid_height - 1, hidden_channels,
                                         kernel_size=deep_supervision_kernel_size, dropout=dropout, input_attn=False,
                                         target_channels_override=deep_supervision_channels_override)

            if deep_supervision_channels_override is not None:
                self.final_conv_small = torch.nn.Conv1d(deep_supervision_channels_override, out_channels, kernel_size=1, bias=False, padding="same", padding_mode="replicate")
                self.final_conv_mid = torch.nn.Conv1d(deep_supervision_channels_override, out_channels, kernel_size=1, bias=False, padding="same", padding_mode="replicate")
                self.final_conv_large = torch.nn.Conv1d(deep_supervision_channels_override, out_channels, kernel_size=1, bias=False, padding="same", padding_mode="replicate")
            else:
                self.final_conv_small = torch.nn.Conv1d(hidden_channels[-3], out_channels, kernel_size=1, bias=False, padding="same", padding_mode="replicate")
                self.final_conv_mid = torch.nn.Conv1d(hidden_channels[-2], out_channels, kernel_size=1, bias=False, padding="same", padding_mode="replicate")
                self.final_conv_large = torch.nn.Conv1d(hidden_channels[-1], out_channels, kernel_size=1, bias=False, padding="same", padding_mode="replicate")

        self.deep_supervision_contraction = deep_supervision_contraction
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
                                                       attention_mode="learned",
                                                       input_length=expected_attn_input_length // (3 * (2 ** (len(blocks) - 2))),
                                                       dropout_pos_embeddings=dropout_pos_embeddings)
            )
        self.no_contraction_head = UnetHead3f(self.pyramid_height - 3, hidden_channels,
                                                kernel_size=1, dropout=dropout)
        self.outconv = torch.nn.Conv1d(stem_final_layer_channels,
                                       attn_out_channels, kernel_size=1,
                                       bias=True, padding="same", padding_mode="replicate")
        self.expected_attn_input_length = expected_attn_input_length
        self.attention_bottleneck = attention_bottleneck

    def call_deep(self, ret, downsampling_methods):
        if self.deep_supervision_contraction:
            x_small = self.small_head(ret[:-2], downsampling_methods)
            x_mid = self.mid_head(ret[:-1], downsampling_methods)
            x_large = self.large_head(ret, downsampling_methods)
        else:
            x_small = self.small_head(ret[:-2])
            x_mid = self.mid_head(ret[:-1])
            x_large = self.large_head(ret)

        x_small = self.final_conv_small(x_small)
        x_mid = self.final_conv_mid(x_mid)
        x_large = self.final_conv_large(x_large)
        return x_small, x_mid, x_large

    def forward(self, x, ret_type="all"):
        assert ret_type in ["all", "deep", "attn"], "ret_type must be one of all, deep, attn"

        N, C, T = x.shape
        # generate list of downsampling methods
        downsampling_methods = [0] * self.pyramid_height
        assert T % self.input_length_multiple == 0, "T must be divisible by {}".format(self.input_length_multiple)

        if ret_type != "deep":
            assert T == self.expected_attn_input_length, "T: {}, self.expected_attn_input_length: {}".format(T, self.expected_attn_input_length)

        # run stem
        ret = self.stem(x, downsampling_methods)

        # final conv
        if ret_type == "deep":
            return self.call_deep(ret, downsampling_methods)
        elif ret_type == "attn":
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
        elif ret_type == "all":
            x_small, x_mid, x_large = self.call_deep(ret, downsampling_methods)

            x = ret[-1] # final layer from conv stem
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
            return x, x_small, x_mid, x_large
