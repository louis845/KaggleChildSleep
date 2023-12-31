import os
import argparse
import json

import numpy as np
import pandas as pd
import torch
import tqdm
import h5py

import manager_folds
import model_event_density_unet
import config
import convert_to_npy_naive

FOLDER = "./inference_density_statistics/density_labels"

def inference(model_dir, validation_entries, all_data,
              hidden_blocks, hidden_channels, bottleneck_factor, kernel_size,
              attention_blocks, upconv_channels_override,
              expand, prediction_length,
              use_anglez_only, use_enmo_only,
              batch_size, use_time_information,
              use_center_softmax,

              use_swa,

              stride_count=24, flip_augmentation=False, use_best_model=False,
              show_tqdm_bar=True,

              out_folder=None):
    print("Using stride count: {}".format(stride_count))
    assert out_folder is not None, "Must specify out_folder."
    assert isinstance(out_folder, str), "out_folder must be a string."

    # init model
    in_channels = 1 if (use_anglez_only or use_enmo_only) else 2
    model = model_event_density_unet.EventDensityUnet(in_channels, hidden_channels,
                                                      kernel_size=kernel_size, blocks=hidden_blocks,
                                                      bottleneck_factor=bottleneck_factor,
                                                      attention_blocks=attention_blocks,
                                                      upconv_channels_override=upconv_channels_override, attention_mode="length",

                                                      use_time_input=use_time_information, training_strategy="density_and_confidence" if use_center_softmax else "density_only",
                                                      input_interval_length=prediction_length, input_expand_radius=expand)
    model = model.to(config.device)

    # load model
    if use_best_model:
        val_metrics = pd.read_csv(os.path.join(model_dir, "val_metrics.csv"), index_col=0)
        val_mAP = val_metrics["val_onset_dense_loc_softmax_mAP"] + val_metrics["val_wakeup_dense_loc_softmax_mAP"] # take with grain of salt, using validation metrics to pick best model
        best_model_idx = int(val_mAP.idxmax())
        if not os.path.isfile(os.path.join(model_dir, "swa_model_{}.pt".format(best_model_idx))):
            use_swa = False
        if use_swa:
            model = torch.optim.swa_utils.AveragedModel(model)
            model.load_state_dict(torch.load(os.path.join(model_dir, "swa_model_{}.pt".format(best_model_idx))))
        else:
            model.load_state_dict(torch.load(os.path.join(model_dir, "model_{}.pt".format(best_model_idx))))
    else:
        if use_swa:
            model = torch.optim.swa_utils.AveragedModel(model)
            model.load_state_dict(torch.load(os.path.join(model_dir, "swa_model.pt")))
        else:
            model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))
    model.eval()

    # inference
    with torch.no_grad():
        if show_tqdm_bar:
            pbar = tqdm.tqdm(total=len(validation_entries))
        for k in range(len(validation_entries)):
            series_id = validation_entries[k]  # series ids

            # create the folder for the series id
            series_folder = os.path.join(out_folder, series_id)
            os.mkdir(series_folder)

            # load the batch
            accel_data = all_data[series_id]["accel"]
            if use_anglez_only:
                accel_data = accel_data[0:1, :]
            elif use_enmo_only:
                accel_data = accel_data[1:2, :]

            # load the times if required
            if use_time_information:
                times = {"hours": all_data[series_id]["hours"], "mins": all_data[series_id]["mins"],
                         "secs": all_data[series_id]["secs"]}
            else:
                times = None


            total_length = accel_data.shape[1]
            probas = np.zeros((2, total_length), dtype=np.float32)
            multiplicities = np.zeros((total_length,), dtype=np.int32)

            intervals_start = [] # list length N
            intervals_end = [] # list length N
            intervals_logits = [] # shape (N, 2, T)
            intervals_event_presence = [] # shape (N, 2)

            for interval_info in model_event_density_unet.event_density_logit_iterator(model=model, time_series=accel_data,
                                                                batch_size=batch_size,
                                                                prediction_length=prediction_length,
                                                                expand=expand, times=times,
                                                                stride_count=stride_count,
                                                                flip_augmentation=flip_augmentation,

                                                                use_time_input=use_time_information,
                                                                device=config.device):
                interval_start = interval_info["interval_start"]
                interval_end = interval_info["interval_end"]
                interval_out_scores = interval_info["interval_out_scores"]
                interval_logits = interval_info["interval_logits"]
                interval_event_presence = interval_info["interval_event_presence"]

                probas[:, interval_start:interval_end] += interval_out_scores
                multiplicities[interval_start:interval_end] += 1

                intervals_start.append(int(interval_start))
                intervals_end.append(int(interval_end))
                intervals_logits.append(interval_logits.astype(np.float16))
                intervals_event_presence.append(interval_event_presence.astype(np.float16))

            multiplicities[multiplicities == 0] = 1
            probas = probas / multiplicities

            # save the probas
            np.save(os.path.join(series_folder, "probas.npy"), probas)

            # save the intervals
            with h5py.File(os.path.join(series_folder, "intervals.h5"), "w") as f:
                intervals_start = np.array(intervals_start, dtype=np.int32)
                intervals_end = np.array(intervals_end, dtype=np.int32)
                intervals_logits = np.stack(intervals_logits, axis=0)
                intervals_event_presence = np.stack(intervals_event_presence, axis=0)

                f.create_dataset("intervals_start", data=intervals_start, dtype=np.int32,
                                        compression="gzip", compression_opts=2)
                f.create_dataset("intervals_end", data=intervals_end, dtype=np.int32,
                                        compression="gzip", compression_opts=2)
                f.create_dataset("intervals_logits", data=intervals_logits, dtype=np.float16,
                                        compression="gzip", compression_opts=5)
                f.create_dataset("intervals_event_presence", data=intervals_event_presence, dtype=np.float16,
                                        compression="gzip", compression_opts=5)

            if show_tqdm_bar:
                pbar.update(1)
        if show_tqdm_bar:
            pbar.close()

if __name__ == "__main__":
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)

    parser = argparse.ArgumentParser("Inference script for models trained on regression events data")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[1, 6, 8, 23, 8], help="Number of hidden 2d blocks for ResNet backbone.")
    parser.add_argument("--hidden_channels", type=int, nargs="+", default=[2], help="Number of hidden channels. Default 2. Can be a list to specify num channels in each downsampled layer.")
    parser.add_argument("--bottleneck_factor", type=int, default=4, help="The bottleneck factor of the ResNet backbone. Default 4.")
    parser.add_argument("--kernel_size", type=int, default=11, help="Kernel size for the first layer. Default 11.")
    parser.add_argument("--attention_blocks", type=int, default=4, help="Number of attention blocks to use. Default 4.")
    parser.add_argument("--upconv_channels_override", type=int, default=None, help="Number of fixed channels for the upsampling path. Default None, do not override.")
    parser.add_argument("--expand", type=int, default=8640, help="Expand the intervals by this amount. Default 8640.")
    parser.add_argument("--use_batch_norm", action="store_true", help="Whether to use batch norm. Default False.")
    parser.add_argument("--prediction_length", type=int, default=17280, help="Number of timesteps to predict. Default 17280.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size. Default 512.")
    parser.add_argument("--use_time_information", action="store_true", help="Whether to use time information. Default False.")
    config.add_argparse_arguments(parser)
    args = parser.parse_args()

    # initialize gpu
    config.parse_args(args)

    # get data parameters
    hidden_blocks = args.hidden_blocks
    hidden_channels = args.hidden_channels
    bottleneck_factor = args.bottleneck_factor
    kernel_size = args.kernel_size
    attention_blocks = args.attention_blocks
    upconv_channels_override = args.upconv_channels_override
    expand = args.expand
    use_batch_norm = args.use_batch_norm
    prediction_length = args.prediction_length
    batch_size = args.batch_size
    use_time_information = args.use_time_information

    args = {
        "hidden_blocks": hidden_blocks,
        "hidden_channels": hidden_channels,
        "bottleneck_factor": bottleneck_factor,
        "kernel_size": kernel_size,
        "attention_blocks": attention_blocks,
        "upconv_channels_override": upconv_channels_override,
        "expand": expand,
        "use_batch_norm": use_batch_norm,
        "prediction_length": prediction_length,
        "batch_size": batch_size,
        "use_time_information": use_time_information,
        "use_center_softmax": False,
        "stride_count": 24,
        "flip_augmentation": False,
        "use_best_model": False,
        "use_swa": False
    }

    # load data
    all_data = convert_to_npy_naive.load_all_data_into_dict()

    # load options
    with open(os.path.join("./inference_density_statistics", "inference_density_preds_options.json"), "r") as f:
        options = json.load(f)

    # inference
    for option in options:
        name = option["name"]
        folder_name = option["folder_name"]
        models = option["models"]
        entries = option["entries"]
        input_type = option["input_type"]

        assert len(models) == len(entries), "Number of models and entries must be the same."
        assert input_type in ["anglez", "enmo", "both"], "Input type must be one of 'anglez', 'enmo', or 'both'."

        use_anglez_only = input_type == "anglez"
        use_enmo_only = input_type == "enmo"

        out_dir = os.path.join(FOLDER, folder_name)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        else:
            continue
        with open(os.path.join(out_dir, "name.txt"), "w") as f:
            f.write(name)

        print("Running inference on {}...".format(name))

        # override args
        opt_args = args.copy()
        for key in option:
            if key not in ["name", "folder_name", "models", "entries", "input_type"]:
                opt_args[key] = option[key]

        for k in range(len(models)):
            model_name = models[k]
            validation_entries = manager_folds.load_dataset(entries[k])
            model_dir = os.path.join("./models", model_name)

            inference(model_dir, validation_entries, all_data,
                                    opt_args["hidden_blocks"], opt_args["hidden_channels"], opt_args["bottleneck_factor"], opt_args["kernel_size"],
                                    opt_args["attention_blocks"], opt_args["upconv_channels_override"],
                                    opt_args["expand"], opt_args["prediction_length"],
                                    use_anglez_only, use_enmo_only,
                                    opt_args["batch_size"], opt_args["use_time_information"],
                                    opt_args["use_center_softmax"],

                                    use_swa=opt_args["use_swa"],

                                    stride_count=opt_args["stride_count"], flip_augmentation=opt_args["flip_augmentation"],
                                    use_best_model=opt_args["use_best_model"],

                                    out_folder=out_dir)
