import torch
from model_unet import *
from model_attention_unet import *

class UnetAttnHead(torch.nn.Module):
    def __init__(self, pyramid_height, hidden_channels, kernel_size,
                 use_batch_norm: bool, dropout: float):
        super(UnetAttnHead, self).__init__()
        assert isinstance(hidden_channels, list), "hidden_channels must be a list"
        for channel in hidden_channels:
            assert isinstance(channel, int), "hidden_channels must be a list of ints"

        self.pyramid_height = pyramid_height

        self.upsample_conv = torch.nn.ModuleList()
        self.upsample_norms = torch.nn.ModuleList()
        for k in range(self.pyramid_height - 1):
            if hidden_channels[k] % 4 == 0:
                num_groups = hidden_channels[k] // 4
                self.upsample_conv.append(torch.nn.Conv1d(hidden_channels[k + 1], hidden_channels[k],
                                                          kernel_size=kernel_size, groups=num_groups, bias=False,
                                                          padding="same", padding_mode="replicate"))
            else:
                self.upsample_conv.append(torch.nn.Conv1d(hidden_channels[k + 1], hidden_channels[k],
                                                          kernel_size=1, bias=False, padding="same",
                                                          padding_mode="replicate"))
            if use_batch_norm:
                self.upsample_norms.append(
                    torch.nn.BatchNorm1d(hidden_channels[k], momentum=BATCH_NORM_MOMENTUM, affine=True))
            else:
                self.upsample_norms.append(torch.nn.InstanceNorm1d(hidden_channels[k]))

        self.cat_conv = torch.nn.ModuleList()
        self.cat_norms = torch.nn.ModuleList()
        for k in range(self.pyramid_height - 1):
            self.cat_conv.append(torch.nn.Conv1d(hidden_channels[k] * 2, hidden_channels[k],
                                                 kernel_size=1, bias=False, padding="same", padding_mode="replicate"))
            if use_batch_norm:
                self.cat_norms.append(
                    torch.nn.BatchNorm1d(hidden_channels[k], momentum=BATCH_NORM_MOMENTUM, affine=True))
            else:
                self.cat_norms.append(torch.nn.InstanceNorm1d(hidden_channels[k]))

        self.nonlin = torch.nn.GELU()
        if dropout > 0.0:
            self.dropout = torch.nn.Dropout1d(dropout)

        self.use_dropout = dropout > 0.0

    def forward(self, ret):
        x = ret[-1]
        for i in range(self.pyramid_height - 1):
            # upsample
            if (i == self.pyramid_height - 2):
                x = self.upsample_3f(x)
            elif i > 0:
                x = self.upsample(x)
            # convolve and normalize
            x = self.upsample_norms[-(i + 1)](self.upsample_conv[-(i + 1)](x))
            x = self.nonlin(x)
            if (i < 2) and self.use_dropout:
                x = self.dropout(x)
            # concatenate
            assert x.shape == ret[-(i + 2)].shape
            x = torch.cat([
                x,
                ret[-(i + 2)]
            ], dim=1)
            # convolve and normalize
            x = self.cat_norms[-(i + 1)](self.cat_conv[-(i + 1)](x))
            x = self.nonlin(x)
            if (i < 2) and self.use_dropout:
                x = self.dropout(x)
        return x

    def upsample(self, x):
        return torch.nn.functional.interpolate(x, size=(x.shape[-1] * 2,), mode="linear")

    def upsample_3f(self, x):
        return torch.nn.functional.interpolate(x, size=(x.shape[-1] * 3,), mode="linear")

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

    def forward(self, x, ret_type="deep"):
        assert ret_type in ["deep"], "ret_type must be one of ['deep']"

        N, C, T = x.shape
        # generate list of downsampling methods
        downsampling_methods = [0] * self.pyramid_height
        assert T % self.input_length_multiple == 0, "T must be divisible by {}".format(self.input_length_multiple)

        ret = self.backbone(x, downsampling_methods)
        x = self.upconv_head(ret)
        x = self.final_conv(x)
        return x

    def get_device(self) -> torch.device:
        return next(self.parameters()).device

class EventConfidenceUnet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, blocks=[4, 6, 8, 8, 8, 8],
                 bottleneck_factor=4, squeeze_excitation=True, squeeze_excitation_bottleneck_factor=4,
                 dropout=0.0, use_batch_norm=False, # settings for stem (backbone)

                 attention_channels=128, attention_heads=4, # settings for attention upsampling module
                 attention_key_query_channels=64, attention_blocks=4,
                 expected_attn_input_length=17280, dropout_pos_embeddings=False,
                 attn_out_channels=1, attention_bottleneck=None,
                 upconv_channels_override=None):
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
        """self.no_contraction_head = UnetAttnHead(self.pyramid_height + 1, hidden_channels + [hidden_channels[-1]],
                                                kernel_size=1, use_batch_norm=True, dropout=dropout)
        self.outconv = torch.nn.Conv1d(hidden_channels[0],
                                       attn_out_channels, kernel_size=1,
                                       bias=True, padding="same", padding_mode="replicate")"""
        self.no_contraction_head = UnetHead3f(self.pyramid_height, hidden_channels,
                                                kernel_size=1, dropout=dropout, input_attn=True, use_batch_norm=use_batch_norm,
                                                target_channels_override=upconv_channels_override)
        if upconv_channels_override is not None:
            self.outconv = torch.nn.Conv1d(upconv_channels_override,
                                           attn_out_channels, kernel_size=1,
                                           bias=True, padding="same", padding_mode="replicate")
        else:
            self.outconv = torch.nn.Conv1d(hidden_channels[-1],
                                           attn_out_channels, kernel_size=1,
                                           bias=True, padding="same", padding_mode="replicate")
        self.expected_attn_input_length = expected_attn_input_length
        self.attention_bottleneck = attention_bottleneck
        self.num_attention_blocks = attention_blocks

    def forward(self, x, ret_type="attn"):
        assert ret_type in ["attn"], "ret_type must be one of ['attn']"

        N, C, T = x.shape
        # generate list of downsampling methods
        downsampling_methods = [0] * self.pyramid_height
        assert T % self.input_length_multiple == 0, "T must be divisible by {}".format(self.input_length_multiple)
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

        if self.num_attention_blocks > 0:
            for i in range(len(self.attention_blocks)):
                x = self.attention_blocks[i](x)

            x = self.no_contraction_head(ret, x)
        else:
            x = self.no_contraction_head(ret, torch.zeros_like(x))
        x = self.outconv(x)
        return x, None, None, None

    def get_device(self) -> torch.device:
        return next(self.parameters()).device

def index_array(arr: np.ndarray, low: int, high: int):
    # index select arr along the last axis, from low to high
    # automatically pad with zeros if low, high are out of bounds
    assert len(arr.shape) == 2, "arr must be a 2D array"
    assert low < high, "low must be less than high"
    assert not ((low < 0 and high <= 0) or (low >= arr.shape[1] and high > arr.shape[1])), "low and high must intersect with arr"

    if 0 <= low < arr.shape[1] and 0 < high <= arr.shape[1]:
        return arr[:, low:high]
    elif low < 0 and 0 < high <= arr.shape[1]:
        return np.concatenate([np.zeros((arr.shape[0], -low), dtype=arr.dtype), arr[:, :high]], axis=1)
    elif 0 <= low < arr.shape[1] and high > arr.shape[1]:
        return np.concatenate([arr[:, low:], np.zeros((arr.shape[0], high - arr.shape[1]), dtype=arr.dtype)], axis=1)
    elif low < 0 and high > arr.shape[1]:
        return np.concatenate([np.zeros((arr.shape[0], -low), dtype=arr.dtype), arr, np.zeros((arr.shape[0], high - arr.shape[1]), dtype=arr.dtype)], axis=1)

def event_regression_inference(model: EventRegressorUnet, time_series: np.ndarray, target_multiple: int=24):
    # target multiple equal to 3 * (2 ** (len(blocks) - 2)) in general
    assert len(time_series.shape) == 2, "time_series must be a 2D array"

    series_length = time_series.shape[1]
    target_length = (series_length // target_multiple) * target_multiple
    start = (series_length - target_length) // 2
    end = start + target_length
    end_contraction = series_length - end
    time_series = time_series[:, start:end]

    time_series_batch = torch.tensor(time_series, dtype=torch.float32, device=model.get_device()).unsqueeze(0)

    # get predictions now
    preds = model(time_series_batch, ret_type="deep")
    preds = preds.squeeze(0)
    preds = preds.cpu().numpy()
    preds = np.pad(preds, ((0, 0), (start, end_contraction)), mode="constant")

    return preds

def event_confidence_inference(model: EventConfidenceUnet, time_series: np.ndarray, batch_size=32,
                               prediction_length=17280, expand=8640):
    model.eval()

    total_length = time_series.shape[1]
    probas = np.zeros((2, total_length), dtype=np.float32)
    multiplicities = np.zeros((total_length,), dtype=np.int32)

    starts = []
    for k in range(4):
        starts.append(k * prediction_length // 4)
    for k in range(4):
        starts.append((total_length + k * prediction_length // 4) % prediction_length)

    for start in starts:
        event_confidence_single_inference(model, time_series, probas, multiplicities,
                                          batch_size=batch_size,
                                          prediction_length=prediction_length, prediction_stride=prediction_length,
                                          expand=expand, start=start)
    multiplicities[multiplicities == 0] = 1 # avoid division by zero. the probas will be zero anyway
    return probas / multiplicities # auto broadcast

def event_confidence_single_inference(model: EventConfidenceUnet, time_series: np.ndarray,
                                      probas: np.ndarray, multiplicities: np.ndarray,
                                      batch_size=32,
                                      prediction_length=17280, prediction_stride=17280,
                                      expand=8640, start=0):
    model.eval()

    assert len(time_series.shape) == 2, "time_series must be a 2D array"
    assert len(probas.shape) == 2, "probas must be a 2D array"
    assert len(multiplicities.shape) == 1, "multiplicities must be a 1D array"

    assert time_series.shape[1] == probas.shape[1], "time_series and probas must have the same length"
    assert time_series.shape[1] == multiplicities.shape[0], "time_series and multiplicities must have the same length"

    assert probas.dtype == np.float32, "probas must be a float32 array"
    assert multiplicities.dtype == np.int32, "multiplicities must be an int32 array"

    total_length = time_series.shape[1]

    # compute batch end points
    batches_starts = []
    batches_ends = []
    while start + prediction_length <= total_length:
        end = start + prediction_length

        batches_starts.append(start)
        batches_ends.append(end)

        start += prediction_stride

    # compute batch predictions
    batches_computed = 0
    while batches_computed < len(batches_starts):
        batches_compute_end = min(batches_computed + batch_size, len(batches_starts))

        # create torch tensor
        batches = []
        for i in range(batches_computed, batches_compute_end):
            batches.append(index_array(time_series, batches_starts[i] - expand, batches_ends[i] + expand))
        batches = np.stack(batches, axis=0)
        batches_torch = torch.tensor(batches, dtype=torch.float32, device=model.get_device())

        # run model
        with torch.no_grad():
            batches_out, _, _, _ = model(batches_torch)
            batches_out = torch.sigmoid(batches_out[:, :, expand:-expand])
            batches_out = batches_out.cpu().numpy()

        # update probas
        for i in range(batches_computed, batches_compute_end):
            k = i - batches_computed
            probas[:, batches_starts[i]:batches_ends[i]] += batches_out[k, :, :]
            multiplicities[batches_starts[i]:batches_ends[i]] += 1

        batches_computed = batches_compute_end
