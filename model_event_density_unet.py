import torch.nn

from model_event_unet import *

def stable_softmax(x: np.ndarray) -> np.ndarray:
    assert isinstance(x, np.ndarray), "x must be a numpy array"
    assert len(x.shape) == 1, "x must be a 1D array"
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

class EventDensityUnet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=[2, 2, 4, 8, 16, 32, 32], kernel_size=3,
                 blocks=[2, 2, 2, 2, 2, 2, 2],
                 bottleneck_factor=4, dropout=0.0,  # settings for stem (backbone)

                 # important input, output settings
                 use_time_input=False, training_strategy="density_only",  # input and training mode
                 input_interval_length=17280, input_expand_radius=8640,  # input settings

                 attention_channels=128, attention_heads=4,  # settings for attention upsampling module
                 attention_key_query_channels=64, attention_blocks=1,
                 upconv_channels_override=8, attention_dropout=0.0,
                 attention_mode="length"):
        super(EventDensityUnet, self).__init__()
        expected_attn_input_length = input_interval_length + 2 * input_expand_radius
        assert kernel_size % 2 == 1, "kernel size must be odd"
        assert len(blocks) >= 6, "blocks must have at least 6 elements"
        assert isinstance(hidden_channels, list), "hidden_channels must be a list"
        assert len(hidden_channels) == len(blocks), "hidden_channels must have the same length as blocks"
        for channel in hidden_channels:
            assert isinstance(channel, int), "hidden_channels must be a list of ints"
        assert training_strategy in ["density_only",
                                     "density_and_confidence"], "training_strategy must be one of ['density_only', 'density_and_confidence']"

        self.input_length_multiples = [1] + [3 * (2 ** k) for k in range(len(blocks) - 1)]
        self.input_length_multiple = self.input_length_multiples[-1]

        # modules for stem and deep supervision outputs
        self.stem = ResNetBackbone(in_channels, hidden_channels, kernel_size, blocks, bottleneck_factor,
                                   squeeze_excitation=False, squeeze_excitation_bottleneck_factor=1, dropout=dropout,
                                   use_batch_norm=True, downsample_3f=True)
        self.pyramid_height = len(blocks)
        self.use_dropout = dropout > 0.0

        # modules for attention and outconv
        stem_final_layer_channels = hidden_channels[-1]

        self.attention_blocks = torch.nn.ModuleList()
        for i in range(attention_blocks):
            self.attention_blocks.append(
                AttentionBlock1DWithPositionalEncoding(channels=stem_final_layer_channels,
                                                       hidden_channels=attention_channels,
                                                       key_query_channels=attention_key_query_channels,
                                                       heads=attention_heads,
                                                       dropout=dropout,
                                                       attention_mode=attention_mode,
                                                       input_length=expected_attn_input_length // self.input_length_multiple,
                                                       dropout_pos_embeddings=False,
                                                       attn_dropout=attention_dropout,
                                                       use_time_input=use_time_input)
            )

        self.no_contraction_head = UnetHead3f(self.pyramid_height, hidden_channels,
                                              kernel_size=1, dropout=dropout, input_attn=True, use_batch_norm=False,
                                              target_channels_override=upconv_channels_override)

        self.presence_conv = torch.nn.Conv1d(stem_final_layer_channels * 2, 8, kernel_size=1, bias=False)
        self.presence_norm = torch.nn.InstanceNorm1d(8, affine=True)
        self.presence_nonlin = torch.nn.GELU()
        self.out_presence_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.out_presence_conv = torch.nn.Conv1d(8, 2, kernel_size=1, bias=True)

        if training_strategy == "density_and_confidence":
            self.out_confidence_conv = torch.nn.Conv1d(stem_final_layer_channels, 2, kernel_size=1, bias=True)

        if upconv_channels_override is not None:
            self.outconv = torch.nn.Conv1d(upconv_channels_override,
                                           2, kernel_size=1,
                                           bias=True, padding="same", padding_mode="replicate")
        else:
            self.outconv = torch.nn.Conv1d(hidden_channels[-1],
                                           2, kernel_size=1,
                                           bias=True, padding="same", padding_mode="replicate")
        self.expected_attn_input_length = expected_attn_input_length
        self.num_attention_blocks = attention_blocks
        self.use_time_input = use_time_input
        self.stem_final_layer_channels = stem_final_layer_channels
        self.training_strategy = training_strategy
        self.input_interval_length = input_interval_length
        self.input_expand_radius = input_expand_radius
        if use_time_input:
            self.period_length = 17280

    def forward(self, x, time: Optional[np.ndarray] = None, return_as_training=False):
        if self.use_time_input:
            assert time is not None, "time must be provided if use_time_input is True"
            assert isinstance(time, np.ndarray), "time must be a numpy array"
            assert time.dtype == np.int32, "time must be a int32 array"
            assert len(time.shape) == 1, "time must be a 1D array"
            assert time.shape[0] == x.shape[0], "time must have the same length as x.shape[0]"
        else:
            assert time is None, "time must be None if use_time_input is False"

        N, C, T = x.shape
        # generate list of downsampling methods
        downsampling_methods = [0] * self.pyramid_height
        assert T % self.input_length_multiple == 0, "T must be divisible by {}".format(self.input_length_multiple)
        assert T == self.expected_attn_input_length, "T: {}, self.expected_attn_input_length: {}".format(T, self.expected_attn_input_length)

        # run stem
        ret = self.stem(x, downsampling_methods)

        x = ret[-1]  # final layer from conv stem
        attn_lvl_expand_radius = self.input_expand_radius // self.input_length_multiple

        if self.num_attention_blocks > 0:
            time_tensor = None
            if self.use_time_input:
                with (torch.no_grad()):
                    time_tensor = 2 * np.pi * torch.tensor(time, dtype=torch.float32,
                                                           device=self.get_device()).unsqueeze(-1).unsqueeze(-1) / self.period_length
                    length_time_tensor = 2 * np.pi * torch.arange(
                        self.expected_attn_input_length // self.input_length_multiple, dtype=torch.float32,
                        device=self.get_device()).unsqueeze(0).unsqueeze(0) / (self.period_length // self.input_length_multiple)
                    channel_time_tensor = 2 * np.pi * torch.arange(self.stem_final_layer_channels // 2,
                                                                   dtype=torch.float32,
                                                                   device=self.get_device()).unsqueeze(0).unsqueeze(-1)\
                                          / (self.stem_final_layer_channels // 2)
                    time_tensor = torch.sin(time_tensor + length_time_tensor + channel_time_tensor)

            for i in range(len(self.attention_blocks)):
                x = self.attention_blocks[i](x, time_tensor)

            attn_out = x
            if self.training_strategy == "density_and_confidence":
                # restrict to middle (discarding expanded parts). this mean we upconv without the expanded parts of the interval
                x = self.no_contraction_head(
                    [ret_tensor[:, :, (self.input_expand_radius // self.input_length_multiples[i]):-(self.input_expand_radius // self.input_length_multiples[i])]\
                                                for i, ret_tensor in enumerate(ret)],
                    x[:, :, attn_lvl_expand_radius:-attn_lvl_expand_radius]
                )
            else:
                x = self.no_contraction_head(ret, x)
        else:
            attn_out = x
            if self.self.training_strategy == "density_and_confidence":
                # restrict to middle (discarding expanded parts). this mean we upconv without the expanded parts of the interval
                x = self.no_contraction_head(
                    [ret_tensor[:, :, (self.input_expand_radius // self.input_length_multiples[i]):-(self.input_expand_radius // self.input_length_multiples[i])] \
                                                for i, ret_tensor in enumerate(ret)],
                    torch.zeros_like(x[:, :, attn_lvl_expand_radius:-attn_lvl_expand_radius])
                )
            else:
                x = self.no_contraction_head(ret, torch.zeros_like(x))

        presence_vector = self.presence_conv(torch.cat([
            ret[-1][:, :, attn_lvl_expand_radius:-attn_lvl_expand_radius],
            attn_out[:, :, attn_lvl_expand_radius:-attn_lvl_expand_radius]
        ], dim=1))
        presence_vector = self.presence_norm(presence_vector)
        presence_vector = self.presence_nonlin(presence_vector)
        presence_vector = self.out_presence_pool(presence_vector)
        event_presence = self.out_presence_conv(presence_vector)

        if self.training or return_as_training:
            if self.training_strategy == "density_only":
                x = self.outconv(x)
                x = torch.permute(x, [0, 2, 1])  # (N, 2, T) -> (N, T, 2)
                event_presence = event_presence.squeeze(-1) # (N, 2, 1) -> (N, 2)
                return x, event_presence
            else:
                x = self.outconv(x)
                x = torch.permute(x, [0, 2, 1])  # (N, 2, T) -> (N, T, 2)
                event_presence = event_presence.squeeze(-1)  # (N, 2, 1) -> (N, 2)
                confidence = self.out_confidence_conv(attn_out) # (N, 2, T')
                return x, event_presence, confidence
        else:
            if self.training_strategy == "density_only":
                out_logits = self.outconv(x[:, :, self.input_expand_radius:-self.input_expand_radius])
                x = torch.softmax(out_logits, dim=-1)
                event_presence_score = torch.sigmoid(event_presence) * 60.0
                x = x * event_presence_score
                return x, out_logits, event_presence_score
            else:
                out_logits = self.outconv(x)
                x = torch.softmax(out_logits, dim=-1)
                event_presence_score = torch.sigmoid(event_presence) * 60.0
                x = x * event_presence_score
                return x, out_logits, event_presence_score

    def get_device(self) -> torch.device:
        return next(self.parameters()).device


def event_density_inference(model: EventDensityUnet, time_series: np.ndarray,
                               predicted_locations: Optional[dict[str, np.ndarray]]=None, batch_size=32,
                               prediction_length=17280, expand=8640, times: Optional[dict[str, np.ndarray]]=None,
                               stride_count=4, flip_augmentation=False, use_time_input=False, device=None):
    model.eval()
    if isinstance(model, EventDensityUnet):
        use_time_input = model.use_time_input
        device = model.get_device()
    else:
        # swa here. we override the parameters
        assert device is not None, "device must be provided if model is not an EventDensityUnet"

    # initialize time input
    if use_time_input:
        assert times is not None, "times must be provided if model uses time input"
        assert "hours" in times and "mins" in times and "secs" in times, "times must contain hours, mins, secs"
        for key in ["hours", "mins", "secs"]:
            assert isinstance(times[key], np.ndarray), f"times[{key}] must be a numpy array"
            assert len(times[key].shape) == 1, f"times[{key}] must be a 1D array"
            assert len(times[key]) == time_series.shape[1], f"times[{key}] must have the same length as time_series"
        assert not flip_augmentation, "rotation_augmentation must be False if model uses time input"
    else:
        assert times is None, "times must not be provided if model does not use time input"

    model.eval()

    total_length = time_series.shape[1]
    probas = np.zeros((2, total_length), dtype=np.float32)
    multiplicities = np.zeros((total_length,), dtype=np.int32)
    if predicted_locations is not None:
        assert set(predicted_locations.keys()) == {"onset", "wakeup"}, "predicted_locations must contain onset and wakeup"
        assert isinstance(predicted_locations["onset"], np.ndarray), "predicted_locations[onset] must be a numpy array"
        assert isinstance(predicted_locations["wakeup"], np.ndarray), "predicted_locations[wakeup] must be a numpy array"
        assert predicted_locations["onset"].dtype == np.int32 or predicted_locations["onset"].dtype == np.int64,\
            "predicted_locations[onset] must be int32. Found type: {}".format(predicted_locations["onset"].dtype)
        assert predicted_locations["wakeup"].dtype == np.int32 or predicted_locations["wakeup"].dtype == np.int64,\
            "predicted_locations[wakeup] must be int32. Found type: {}".format(predicted_locations["wakeup"].dtype)

        assert np.all(predicted_locations["onset"][1:] > predicted_locations["onset"][:-1]), "predicted_locations[onset] must be sorted"
        assert np.all(predicted_locations["wakeup"][1:] > predicted_locations["wakeup"][:-1]), "predicted_locations[wakeup] must be sorted"

        onset_locs_probas = np.zeros((len(predicted_locations["onset"]),), dtype=np.float32)
        wakeup_locs_probas = np.zeros((len(predicted_locations["wakeup"]),), dtype=np.float32)
        onset_locs_multiplicities = np.zeros((len(predicted_locations["onset"]),), dtype=np.int32)
        wakeup_locs_multiplicities = np.zeros((len(predicted_locations["wakeup"]),), dtype=np.int32)
    else:
        onset_locs_probas, onset_locs_multiplicities = None, None
        wakeup_locs_probas, wakeup_locs_multiplicities = None, None

    starts = []
    for k in range(stride_count):
        starts.append(k * prediction_length // stride_count)
    for k in range(stride_count):
        starts.append((total_length + k * prediction_length // stride_count) % prediction_length)

    for start in starts:
        event_density_single_inference(model, time_series, probas, multiplicities,
                                          predicted_locations,
                                          onset_locs_probas, onset_locs_multiplicities,
                                          wakeup_locs_probas, wakeup_locs_multiplicities,
                                          device=device, use_time_input=use_time_input,
                                          batch_size=batch_size,
                                          prediction_length=prediction_length, prediction_stride=prediction_length,
                                          expand=expand, start=start, times=times)
        if flip_augmentation:
            event_density_single_inference(model, time_series, probas, multiplicities,
                                              predicted_locations,
                                              onset_locs_probas, onset_locs_multiplicities,
                                              wakeup_locs_probas, wakeup_locs_multiplicities,
                                              device=device, use_time_input=use_time_input,
                                              batch_size=batch_size,
                                              prediction_length=prediction_length, prediction_stride=prediction_length,
                                              expand=expand, start=start, times=times, flipped=True)
    multiplicities[multiplicities == 0] = 1 # avoid division by zero. the probas will be zero anyway

    if predicted_locations is not None:
        if len(predicted_locations["onset"]) > 0:
            onset_locs_multiplicities[onset_locs_multiplicities == 0] = 1
            onset_locs_probas /= onset_locs_multiplicities
        if len(predicted_locations["wakeup"]) > 0:
            wakeup_locs_multiplicities[wakeup_locs_multiplicities == 0] = 1
            wakeup_locs_probas /= wakeup_locs_multiplicities
    return probas / multiplicities, onset_locs_probas, wakeup_locs_probas

def event_density_single_inference(model: EventDensityUnet, time_series: np.ndarray,
                                      probas: np.ndarray, multiplicities: np.ndarray,
                                      predicted_locations: Optional[dict[str, np.ndarray]],
                                      onset_locs_probas: Optional[np.ndarray], onset_locs_multiplicities: Optional[np.ndarray],
                                      wakeup_locs_probas: Optional[np.ndarray], wakeup_locs_multiplicities: Optional[np.ndarray],
                                      device: torch.device, use_time_input: bool,
                                      batch_size=32,
                                      prediction_length=17280, prediction_stride=17280,
                                      expand=8640, start=0, times=None,
                                      flipped=False):
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
        batches_torch = torch.tensor(batches, dtype=torch.float32, device=device)

        # create time ndarray
        if use_time_input:
            batch_times = []
            for i in range(batches_computed, batches_compute_end):
                hour = times["hours"][batches_starts[i]]
                minute = times["mins"][batches_starts[i]]
                second = times["secs"][batches_starts[i]]
                time = (hour * 3600 + minute * 60 + second) // 5
                time -= expand
                if time < 0:
                    time += 17280
                time = time % 17280
                batch_times.append(time)
            batch_times = np.array(batch_times, dtype=np.int32)
        else:
            batch_times = None

        # run model
        with torch.no_grad():
            if flipped:
                batches_out, batches_out_raw_logits, batches_out_event_presence_score = model(torch.flip(batches_torch, dims=[-1,]), time=batch_times)
                batches_out = torch.flip(batches_out, dims=[-1,])
                batches_out_raw_logits = torch.flip(batches_out_raw_logits, dims=[-1,])
            else:
                batches_out, batches_out_raw_logits, batches_out_event_presence_score = model(batches_torch, time=batch_times)

            batches_out = batches_out.cpu().numpy()
            if predicted_locations is not None:
                batches_out_raw_logits = batches_out_raw_logits.cpu().numpy()
                batches_out_event_presence_score = batches_out_event_presence_score.cpu().numpy()

        # update probas
        for i in range(batches_computed, batches_compute_end):
            k = i - batches_computed
            probas[:, batches_starts[i]:batches_ends[i]] += batches_out[k, :, :]
            multiplicities[batches_starts[i]:batches_ends[i]] += 1

            if predicted_locations is not None:
                onset_locations = predicted_locations["onset"]
                wakeup_locations = predicted_locations["wakeup"]

                if len(onset_locations) > 0:
                    onset_locs_idxs = np.searchsorted(onset_locations, [batches_starts[i], batches_ends[i]], side="left")
                    batch_event_presence = batches_out_event_presence_score[k, 0]
                    if onset_locs_idxs[0] < onset_locs_idxs[1] - 1:
                        onset_locs_of_interest = onset_locations[onset_locs_idxs[0]:onset_locs_idxs[1]] - batches_starts[i]
                        batches_out_logits_of_interest = batches_out_raw_logits[k, 0, onset_locs_of_interest]
                        batches_out_probas_of_interest = stable_softmax(batches_out_logits_of_interest)

                        onset_locs_probas[onset_locs_idxs[0]:onset_locs_idxs[1]] += batches_out_probas_of_interest * batch_event_presence
                        onset_locs_multiplicities[onset_locs_idxs[0]:onset_locs_idxs[1]] += 1
                    elif onset_locs_idxs[0] == onset_locs_idxs[1] - 1:
                        onset_locs_probas[onset_locs_idxs[0]] += batch_event_presence
                        onset_locs_multiplicities[onset_locs_idxs[0]] += 1

                if len(wakeup_locations) > 0:
                    wakeup_locs_idxs = np.searchsorted(wakeup_locations, [batches_starts[i], batches_ends[i]], side="left")
                    batch_event_presence = batches_out_event_presence_score[k, 1]
                    if wakeup_locs_idxs[0] < wakeup_locs_idxs[1] - 1:
                        wakeup_locs_of_interest = wakeup_locations[wakeup_locs_idxs[0]:wakeup_locs_idxs[1]] - batches_starts[i]
                        batches_out_logits_of_interest = batches_out_raw_logits[k, 1, wakeup_locs_of_interest]
                        batches_out_probas_of_interest = stable_softmax(batches_out_logits_of_interest)

                        wakeup_locs_probas[wakeup_locs_idxs[0]:wakeup_locs_idxs[1]] += batches_out_probas_of_interest * batch_event_presence
                        wakeup_locs_multiplicities[wakeup_locs_idxs[0]:wakeup_locs_idxs[1]] += 1
                    elif wakeup_locs_idxs[0] == wakeup_locs_idxs[1] - 1:
                        wakeup_locs_probas[wakeup_locs_idxs[0]] += batch_event_presence
                        wakeup_locs_multiplicities[wakeup_locs_idxs[0]] += 1


        batches_computed = batches_compute_end

class ProbasDilationConverter:
    def __init__(self, sigma: int, device: torch.device):
        """
        Class for converting probas to dilated probas (akin to adding Gaussian random noise).
        """
        self.sigma = sigma
        self.device = device

        self.conv = torch.nn.Conv1d(1, 1, kernel_size=10 * sigma + 1,
                                                 bias=False, padding="same", padding_mode="zeros")
        # initialize weights
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor(
                np.exp(-np.arange(-5 * sigma, 5 * sigma + 1) ** 2),
            dtype=torch.float32, device="cpu"))
        self.conv.to(device)

    def convert(self, probas: np.ndarray):
        """
        Convert probas to IOU probas scores.
        :param probas: The probas to convert
        :return: The converted probas
        """
        assert len(probas.shape) == 1, "probas must be a 1D array"

        with torch.no_grad():
            probas_torch = torch.tensor(probas, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            dilated_probas_torch = self.conv(probas_torch).squeeze(0).squeeze(0)
            dilated_probas = dilated_probas_torch.cpu().numpy()

        return dilated_probas
