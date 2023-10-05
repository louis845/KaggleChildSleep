import torch
import numpy as np

class ResConv1DBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 downsample, bottleneck_factor=1,
                 squeeze_excitation=False,
                 squeeze_excitation_bottleneck_factor=4,
                 kernel_size=11,
                 dropout=0.0):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param downsample: whether to downsample the input.
        :param bottleneck_factor: factor by which to reduce the number of channels in the bottleneck
        :param squeeze_excitation: whether to use squeeze and excitation
        :param squeeze_excitation_bottleneck_factor: factor by which to reduce the number of channels in the squeeze and excitation block
        :param kernel_size: kernel size of the convolutional layers
        """
        super(ResConv1DBlock, self).__init__()
        assert in_channels <= out_channels
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        if bottleneck_factor > 1:
            assert out_channels % bottleneck_factor == 0, "out_channels must be divisible by bottleneck_factor"
            assert out_channels % (bottleneck_factor * 4) == 0, "out_channels must be divisible by bottleneck_factor * 4"
        assert 0.0 <= dropout <= 1.0, "dropout must be between 0.0 and 1.0"
        bottleneck_channels = out_channels // bottleneck_factor
        use_dropout = dropout > 0.0

        if downsample:
            self.avgpool = torch.nn.AvgPool1d(kernel_size=2, stride=2, padding=0)

        if bottleneck_factor > 1:
            self.conv1 = torch.nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False,
                                         padding="same", padding_mode="replicate")
            self.batchnorm1 = torch.nn.InstanceNorm1d(bottleneck_channels, affine=True)
            self.nonlin1 = torch.nn.GELU()
            if use_dropout:
                self.dropout1 = torch.nn.Dropout1d(dropout)

            num_groups = bottleneck_channels // 4
            if downsample:
                self.conv2 = torch.nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=kernel_size * 2, stride=2,
                                             bias=False, padding=kernel_size - 1, padding_mode="replicate", groups=num_groups)
            else:
                self.conv2 = torch.nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=kernel_size, bias=False,
                                             padding="same", padding_mode="replicate", groups=num_groups)
            self.batchnorm2 = torch.nn.InstanceNorm1d(bottleneck_channels, affine=True)
            self.nonlin2 = torch.nn.GELU()
            if use_dropout:
                self.dropout2 = torch.nn.Dropout1d(dropout)

            self.conv3 = torch.nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1, bias=False,
                                         padding="same", padding_mode="replicate")
            self.batchnorm3 = torch.nn.InstanceNorm1d(out_channels, affine=True)
            self.nonlin3 = torch.nn.GELU()
            if use_dropout:
                self.dropout3 = torch.nn.Dropout1d(dropout)
        else:
            self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=False,
                                         padding="same", padding_mode="replicate")
            self.batchnorm1 = torch.nn.InstanceNorm1d(out_channels, affine=True)
            self.nonlin1 = torch.nn.GELU()
            if use_dropout:
                self.dropout1 = torch.nn.Dropout1d(dropout)

            if downsample:
                self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size * 2,
                                             stride=2, padding=kernel_size - 1, padding_mode="replicate", bias=False)
            else:
                self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                                             bias=False, padding="same", padding_mode="replicate")
            self.batchnorm2 = torch.nn.InstanceNorm1d(out_channels, affine=True)
            self.nonlin2 = torch.nn.GELU()
            if use_dropout:
                self.dropout2 = torch.nn.Dropout1d(dropout)
        if squeeze_excitation:
            assert out_channels % squeeze_excitation_bottleneck_factor == 0, "out_channels must be divisible by squeeze_excitation_bottleneck_factor"
            self.se_pool = torch.nn.AdaptiveAvgPool1d(1)
            self.se_conv1 = torch.nn.Conv1d(out_channels, out_channels // squeeze_excitation_bottleneck_factor,
                                            kernel_size=1, bias=True, padding="same", padding_mode="replicate")
            self.se_relu = torch.nn.ReLU()
            self.se_conv2 = torch.nn.Conv1d(out_channels // squeeze_excitation_bottleneck_factor, out_channels,
                                            kernel_size=1, bias=True, padding="same", padding_mode="replicate")
            self.se_sigmoid = torch.nn.Sigmoid()

        self.bottleneck_factor = bottleneck_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.squeeze_excitation = squeeze_excitation
        self.use_dropout = use_dropout

    def forward(self, x, downsampling_method: int):
        # downsampling_method must be 0, 1, 2, where 0 means that we expect the input to be even (or no downsampling),
        # 1 means we expect odd input and pad on left, 2 means we expect odd input and pad on right.
        # if no downsampling is necessary this is ignored
        N, C, T = x.shape
        assert C == self.in_channels
        if self.downsample:
            if downsampling_method == 0:
                assert T % 2 == 0, "T must be even"
            elif downsampling_method == 1:
                assert T % 2 == 1, "T must be odd"
            elif downsampling_method == 2:
                assert T % 2 == 1, "T must be odd"

        if self.in_channels < self.out_channels:
            x_init = torch.nn.functional.pad(x, (
            0, 0, 0, self.out_channels - self.in_channels), "constant", 0.0)
        else:
            x_init = x

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.nonlin1(x)
        if self.use_dropout:
            x = self.dropout1(x)
        assert x.shape == (N, self.out_channels // self.bottleneck_factor, T)

        if self.downsample:
            if downsampling_method == 1:
                x = torch.nn.functional.pad(x, (1, 0), "replicate")
            elif downsampling_method == 2:
                x = torch.nn.functional.pad(x, (0, 1), "replicate")
            x = self.conv2(x)
            if downsampling_method == 0:
                assert x.shape == (N, self.out_channels // self.bottleneck_factor, T // 2)
            else:
                assert x.shape == (N, self.out_channels // self.bottleneck_factor, (T + 1) // 2)
        else:
            x = self.conv2(x)
            assert x.shape == (N, self.out_channels // self.bottleneck_factor, T)
        x = self.batchnorm2(x)

        if self.bottleneck_factor > 1:
            x = self.nonlin2(x)
            if self.use_dropout:
                x = self.dropout2(x)

            x = self.conv3(x)
            x = self.batchnorm3(x)

        if self.squeeze_excitation:
            x_se = self.se_pool(x)
            x_se = self.se_conv1(x_se)
            x_se = self.se_relu(x_se)
            x_se = self.se_conv2(x_se)
            x_se = self.se_sigmoid(x_se)
            x = x * x_se

        if self.downsample:
            if downsampling_method == 1:
                x_init = torch.nn.functional.pad(x_init, (1, 0), "replicate")
            else:
                x_init = torch.nn.functional.pad(x_init, (0, 1), "replicate")
            x_init = self.avgpool(x_init)
            if downsampling_method == 0:
                assert x_init.shape == (N, self.out_channels, T // 2)
            else:
                assert x_init.shape == (N, self.out_channels, (T + 1) // 2)
        if self.bottleneck_factor > 1:
            result = self.nonlin3(x_init + x)
            if self.use_dropout:
                result = self.dropout3(result)
        else:
            result = self.nonlin2(x_init + x)
            if self.use_dropout:
                result = self.dropout2(result)
        return result


class ResConv1D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, downsample=False,
                 bottleneck_factor: int=1, squeeze_excitation=False, squeeze_excitation_bottleneck_factor: int=4,
                 kernel_size=11, dropout=0.0):
        super(ResConv1D, self).__init__()
        assert in_channels <= out_channels
        if out_channels <= 32:
            bottleneck_factor = 1
            dropout = 0.0

        self.conv_res = torch.nn.ModuleList()
        self.conv_res.append(ResConv1DBlock(in_channels, out_channels, downsample=downsample,
                                            bottleneck_factor=bottleneck_factor, squeeze_excitation=squeeze_excitation,
                                            squeeze_excitation_bottleneck_factor=squeeze_excitation_bottleneck_factor,
                                            kernel_size=kernel_size, dropout=dropout))
        for k in range(1, num_blocks):
            self.conv_res.append(ResConv1DBlock(out_channels, out_channels, downsample=False,
                                                bottleneck_factor=bottleneck_factor,
                                                squeeze_excitation=squeeze_excitation,
                                                squeeze_excitation_bottleneck_factor=squeeze_excitation_bottleneck_factor,
                                                kernel_size=kernel_size, dropout=dropout))

        self.num_blocks = num_blocks

    def forward(self, x, downsampling_method: int):
        for k in range(self.num_blocks):
            x = self.conv_res[k](x, downsampling_method)

        return x


class ResNetBackbone(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, blocks=[2, 3, 4, 6, 10, 15, 15],
                 bottleneck_factor=4, squeeze_excitation=True, squeeze_excitation_bottleneck_factor=4,
                 dropout=0.0):
        super(ResNetBackbone, self).__init__()
        assert kernel_size % 2 == 1, "kernel size must be odd"
        self.pyramid_height = len(blocks)

        self.conv_down = torch.nn.ModuleList()
        self.initial_conv = torch.nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size,
                                            bias=False, padding="same", padding_mode="replicate")
        self.initial_batch_norm = torch.nn.InstanceNorm1d(hidden_channels)
        self.initial_nonlin = torch.nn.GELU()

        self.conv_down.append(ResConv1D(hidden_channels, hidden_channels, blocks[0], downsample=False,
                                   bottleneck_factor=bottleneck_factor, squeeze_excitation=squeeze_excitation,
                                   squeeze_excitation_bottleneck_factor=squeeze_excitation_bottleneck_factor,
                                   kernel_size=kernel_size, dropout=dropout))
        for i in range(self.pyramid_height - 1):
            self.conv_down.append(ResConv1D(hidden_channels * (2 ** i), hidden_channels * (2 ** (i + 1)),
                                            blocks[i + 1], downsample=True, bottleneck_factor=bottleneck_factor,
                                            squeeze_excitation=squeeze_excitation,
                                            squeeze_excitation_bottleneck_factor=squeeze_excitation_bottleneck_factor,
                                            kernel_size=kernel_size, dropout=dropout))

    def forward(self, x, downsampling_method: list[int]):
        assert len(downsampling_method) == self.pyramid_height, "downsampling method must be specified for each level"
        x = self.initial_conv(x)
        x = self.initial_batch_norm(x)
        x = self.initial_nonlin(x)

        # contracting path
        ret = []
        for i in range(self.pyramid_height):
            x = self.conv_down[i](x, downsampling_method[i])
            ret.append(x)

        return ret

class Unet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, blocks=[2, 3, 4, 6, 10, 15, 15],
                 bottleneck_factor=4, squeeze_excitation=True, squeeze_excitation_bottleneck_factor=4,
                 odd_random_shift_training=True, dropout=0.0,
                 out_channels=1):
        super(Unet, self).__init__()
        assert kernel_size % 2 == 1, "kernel size must be odd"
        self.backbone = ResNetBackbone(in_channels, hidden_channels, kernel_size, blocks, bottleneck_factor,
                                       squeeze_excitation, squeeze_excitation_bottleneck_factor, dropout=dropout)
        self.pyramid_height = len(blocks)

        self.upsample_conv = torch.nn.ModuleList()
        self.upsample_norms = torch.nn.ModuleList()
        for k in range(self.pyramid_height - 1):
            if (hidden_channels * (2 ** k)) % 4 == 0:
                num_groups = (hidden_channels * (2 ** k)) // 4
                self.upsample_conv.append(torch.nn.Conv1d(hidden_channels * (2 ** (k + 1)), hidden_channels * (2 ** k),
                                                          kernel_size=kernel_size, groups=num_groups, bias=False,
                                                          padding="same", padding_mode="replicate"))
            else:
                self.upsample_conv.append(torch.nn.Conv1d(hidden_channels * (2 ** (k + 1)), hidden_channels * (2 ** k),
                                                          kernel_size=1, bias=False, padding="same", padding_mode="replicate"))
            self.upsample_norms.append(torch.nn.InstanceNorm1d(hidden_channels * (2 ** k)))

        self.cat_conv = torch.nn.ModuleList()
        self.cat_norms = torch.nn.ModuleList()
        for k in range(self.pyramid_height - 1):
            self.cat_conv.append(torch.nn.Conv1d(hidden_channels * (2 ** (k + 1)), hidden_channels * (2 ** k),
                                                 kernel_size=1, bias=False, padding="same", padding_mode="replicate"))
            self.cat_norms.append(torch.nn.InstanceNorm1d(hidden_channels * (2 ** k)))

        self.nonlin = torch.nn.GELU()
        if dropout > 0.0:
            self.dropout = torch.nn.Dropout1d(dropout)

        self.final_conv = torch.nn.Conv1d(hidden_channels, out_channels, kernel_size=kernel_size, bias=False, padding="same", padding_mode="replicate")

        self.odd_random_shift_training = odd_random_shift_training
        self.use_dropout = dropout > 0.0

    def upsample(self, x, downsampling_method):
        if downsampling_method == 0:
            return torch.nn.functional.interpolate(x, size=(x.shape[-1] * 2,), mode="linear")
        elif downsampling_method == 1:
            return torch.nn.functional.interpolate(x, size=(x.shape[-1] * 2,), mode="linear")[..., 1:]
        elif downsampling_method == 2:
            return torch.nn.functional.interpolate(x, size=(x.shape[-1] * 2,), mode="linear")[..., :-1]
        else:
            raise ValueError("downsampling method not supported")

    def forward(self, x):
        N, C, T = x.shape
        # generate list of downsampling methods
        downsampling_methods = [0]
        cur_length = T
        if self.training and self.odd_random_shift_training:
            for k in range(self.pyramid_height - 1):
                if cur_length % 2 == 0:
                    downsampling_methods.append(0)
                    cur_length = cur_length // 2
                else:
                    if np.random.rand() > 0.5:
                        downsampling_methods.append(1)
                    else:
                        downsampling_methods.append(2)
                    cur_length = (cur_length + 1) // 2
        else:
            for k in range(self.pyramid_height - 1):
                if cur_length % 2 == 0:
                    downsampling_methods.append(0)
                    cur_length = cur_length // 2
                else:
                    downsampling_methods.append(1)
                    cur_length = (cur_length + 1) // 2

        # run backbone
        ret = self.backbone(x, downsampling_methods)

        # upsampling path
        x = ret[-1]
        for i in range(self.pyramid_height - 1):
            # upsample
            x = self.upsample(x, downsampling_methods[-(i + 1)])
            # convolve and normalize
            x = self.upsample_norms[-(i + 1)](self.upsample_conv[-(i + 1)](x))
            x = self.nonlin(x)
            if (i < 2) and self.use_dropout:
                x = self.dropout(x)
            # concatenate
            x = torch.cat([
                x,
                ret[-(i + 2)]
            ], dim=1)
            # convolve and normalize
            x = self.cat_norms[-(i + 1)](self.cat_conv[-(i + 1)](x))
            x = self.nonlin(x)
            if (i < 2) and self.use_dropout:
                x = self.dropout(x)

        # final conv
        x = self.final_conv(x)
        return x

