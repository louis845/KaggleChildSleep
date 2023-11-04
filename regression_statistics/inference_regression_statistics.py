import os
import argparse
import json
import sys

import torch
import tqdm
import pandas as pd
from PySide2.QtWidgets import QApplication

import manager_folds
import model_attention_unet
import model_event_unet
import config
import convert_to_npy_naive
import training_resnet_regression
import inference_regression_statistics_visualization

def load_all_events():
    all_events = {}
    events_df = pd.read_csv("./data/train_events.csv")
    events_df = events_df.dropna()

    for series_id in os.listdir("data_naive"):
        series_events = events_df.loc[events_df["series_id"] == series_id]
        if len(series_events) == 0:
            continue

        all_events[series_id] = []

        for night in series_events["night"].unique():
            night_events = series_events.loc[series_events["night"] == night]
            if len(night_events) == 2:
                assert night_events.iloc[0]["event"] == "onset"
                assert night_events.iloc[1]["event"] == "wakeup"
                onset = int(night_events.iloc[0]["step"])
                wakeup = int(night_events.iloc[1]["step"])
                all_events[series_id].append({
                    "onset": onset,
                    "wakeup": wakeup
                })

    return all_events

def obtain_statistics(model_path, entry, regression_width, is_regression=True, is_huber_regression=False, is_standard=False):
    in_channels = 1 if (use_anglez_only or use_enmo_only) else 2
    if is_standard:
        model = model_event_unet.EventRegressorUnet()
    else:
        model = model_attention_unet.Unet3fDeepSupervision(in_channels, hidden_channels, kernel_size=kernel_size,
                                                           blocks=hidden_blocks,
                                                           bottleneck_factor=bottleneck_factor, squeeze_excitation=False,
                                                           squeeze_excitation_bottleneck_factor=4,
                                                           dropout=0.0,
                                                           use_batch_norm=True, out_channels=2, attn_out_channels=2,

                                                           deep_supervision_contraction=not disable_deep_upconv_contraction,
                                                           deep_supervision_kernel_size=deep_upconv_kernel,
                                                           deep_supervision_channels_override=deep_upconv_channels_override)
    model = model.to(config.device)
    model.eval()

    # load model
    model.load_state_dict(torch.load(model_path))

    # load data
    validation_entries = manager_folds.load_dataset(entry)

    # inference
    onset_small_aes, onset_mid_aes, onset_large_aes = [], [], []
    wakeup_small_aes, wakeup_mid_aes, wakeup_large_aes = [], [], []
    onset_all_aes = [onset_small_aes, onset_mid_aes, onset_large_aes]
    wakeup_all_aes = [wakeup_small_aes, wakeup_mid_aes, wakeup_large_aes]
    with torch.no_grad():
        with tqdm.tqdm(total=len(validation_entries)) as pbar:
            for k in range(len(validation_entries)):
                series_id = validation_entries[k]  # series ids
                if series_id not in all_events:
                    continue

                # load the batch
                accel_data = all_data[series_id]["accel"]
                series_length = accel_data.shape[1]
                target_length = (series_length // target_multiple) * target_multiple
                start = (series_length - target_length) // 2
                end = start + target_length
                accel_data = accel_data[:, start:end]

                accel_data_batch = torch.tensor(accel_data, dtype=torch.float32, device=config.device).unsqueeze(0)
                if use_anglez_only:
                    accel_data_batch = accel_data_batch[:, 0:1, :]
                elif use_enmo_only:
                    accel_data_batch = accel_data_batch[:, 1:2, :]

                # get predictions now
                if is_standard:
                    pred = model(accel_data_batch, ret_type="deep")
                    pred = pred.squeeze(0)

                    preds = [pred, pred, pred]
                    preds_np = [pred.cpu().numpy() for pred in preds]
                else:
                    pred_small, pred_mid, pred_large = model(accel_data_batch, ret_type="deep")
                    if not is_regression:
                        pred_small, pred_mid, pred_large = torch.sigmoid(pred_small), torch.sigmoid(pred_mid), torch.sigmoid(pred_large)
                    pred_small, pred_mid, pred_large = pred_small.squeeze(0), pred_mid.squeeze(0), pred_large.squeeze(0)


                    preds = [pred_small, pred_mid, pred_large]
                    preds_np = [pred.cpu().numpy() for pred in preds]


                # predict with events
                for i in range(len(regression_width)):
                    width = int(regression_width[i])
                    for event in all_events[series_id]:
                        onset = event["onset"] - start
                        wakeup = event["wakeup"] - start

                        if is_regression:
                            onset_argmax_ae = training_resnet_regression.local_argmax_kernel_loss(preds[i][0, :], onset, width, use_L1_kernel=is_huber_regression)
                            wakeup_argmax_ae = training_resnet_regression.local_argmax_kernel_loss(preds[i][1, :], wakeup, width, use_L1_kernel=is_huber_regression)
                        else:
                            onset_argmax_ae = training_resnet_regression.local_argmax_loss(preds_np[i][0, :], onset, width)
                            wakeup_argmax_ae = training_resnet_regression.local_argmax_loss(preds_np[i][1, :], wakeup, width)

                        onset_all_aes[i].append(int(onset_argmax_ae))
                        wakeup_all_aes[i].append(int(wakeup_argmax_ae))

                pbar.update(1)

    return onset_small_aes, onset_mid_aes, onset_large_aes, wakeup_small_aes, wakeup_mid_aes, wakeup_large_aes



if __name__ == "__main__":
    FOLDER = "./regression_statistics/generated_statistics"
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)

    # argparse params
    parser = argparse.ArgumentParser("Inference script for models trained on regression events data")
    parser.add_argument("--epochs_per_save", type=int, default=2, help="Number of epochs between saves. Default 2.")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[1, 6, 8, 23, 8],
                        help="Number of hidden 2d blocks for ResNet backbone.")
    parser.add_argument("--hidden_channels", type=int, nargs="+", default=[2],
                        help="Number of hidden channels. Default 2. Can be a list to specify num channels in each downsampled layer.")
    parser.add_argument("--bottleneck_factor", type=int, default=4,
                        help="The bottleneck factor of the ResNet backbone. Default 4.")
    parser.add_argument("--kernel_size", type=int, default=11, help="Kernel size for the first layer. Default 11.")
    parser.add_argument("--disable_deep_upconv_contraction", action="store_true",
                        help="Whether to disable the deep upconv contraction. Default False.")
    parser.add_argument("--deep_upconv_kernel", type=int, default=5,
                        help="Kernel size for the deep upconv layers. Default 5.")
    parser.add_argument("--deep_upconv_channels_override", type=int, default=None,
                        help="Override the number of channels for the deep upconv layers. Default None.")
    parser.add_argument("--use_anglez_only", action="store_true", help="Whether to use only anglez. Default False.")
    parser.add_argument("--use_enmo_only", action="store_true", help="Whether to use only enmo. Default False.")
    config.add_argparse_arguments(parser)
    args = parser.parse_args()

    # load options for statistics computation
    with open("./regression_statistics/inference_regression_stats_options.json", "r") as f:
        options = json.load(f)

    # initialize gpu
    config.parse_args(args)

    # obtain model and training parameters
    hidden_blocks = args.hidden_blocks
    hidden_channels = args.hidden_channels
    bottleneck_factor = args.bottleneck_factor
    kernel_size = args.kernel_size
    disable_deep_upconv_contraction = args.disable_deep_upconv_contraction
    deep_upconv_kernel = args.deep_upconv_kernel
    deep_upconv_channels_override = args.deep_upconv_channels_override
    use_anglez_only = args.use_anglez_only
    use_enmo_only = args.use_enmo_only

    assert not (use_anglez_only and use_enmo_only), "Cannot use both anglez only and enmo only."

    # load all data
    all_data = convert_to_npy_naive.load_all_data_into_dict()
    all_events = load_all_events()
    target_multiple = 3 * (2 ** (len(hidden_blocks) - 2))

    # compute statistics
    all_stats = {}
    regression_names = []
    for k in range(len(options)):
        name = options[k]["name"]
        models = options[k]["models"]
        entries = options[k]["entries"]
        is_regression = options[k]["is_regression"]
        is_huber_regression = "is_huber_regression" in options[k]
        is_standard = options[k]["is_standard"]
        regression_width = options[k]["regression_width"]
        assert len(models) == len(entries), "Number of models and entries must be the same."
        assert isinstance(is_regression, bool), "is_regression must be a boolean."
        if is_standard:
            assert is_regression, "is_standard must be True only if is_regression is True."
        if is_huber_regression:
            assert is_regression, "is_huber_regression must be True only if is_regression is True."

        print("Computing statistics for {}".format(name))
        onset_small_aes, onset_mid_aes, onset_large_aes, wakeup_small_aes, wakeup_mid_aes, wakeup_large_aes = [], [], [], [], [], []
        for j in range(len(models)):
            model_dir = models[j]
            model_path = os.path.join("models", model_dir, "model.pt")
            entry = entries[j]

            result_onset_small_aes, result_onset_mid_aes, result_onset_large_aes,\
                result_wakeup_small_aes, result_wakeup_mid_aes, result_wakeup_large_aes\
                    = obtain_statistics(model_path, entry, is_regression=is_regression, is_huber_regression=is_huber_regression,
                                        is_standard=is_standard, regression_width=regression_width)

            onset_small_aes.extend(result_onset_small_aes)
            onset_mid_aes.extend(result_onset_mid_aes)
            onset_large_aes.extend(result_onset_large_aes)
            wakeup_small_aes.extend(result_wakeup_small_aes)
            wakeup_mid_aes.extend(result_wakeup_mid_aes)
            wakeup_large_aes.extend(result_wakeup_large_aes)

        all_stats[name] = {
            "onset_small_aes": onset_small_aes,
            "onset_mid_aes": onset_mid_aes,
            "onset_large_aes": onset_large_aes,
            "wakeup_small_aes": wakeup_small_aes,
            "wakeup_mid_aes": wakeup_mid_aes,
            "wakeup_large_aes": wakeup_large_aes
        }
        regression_names.append(name)

    # save statistics
    with open(os.path.join(FOLDER, "inference_regression_statistics.json"), "w") as f:
        json.dump(all_stats, f, indent=4)

    # visualize statistics
    app = QApplication(sys.argv)
    w = inference_regression_statistics_visualization.ApplicationWindow(all_stats, regression_names,
                                                                        ["onset_small_aes", "onset_mid_aes", "onset_large_aes",
                                                                            "wakeup_small_aes", "wakeup_mid_aes", "wakeup_large_aes"])
    w.show()
    sys.exit(app.exec_())
