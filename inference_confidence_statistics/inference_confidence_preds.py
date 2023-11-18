import os
import argparse
import json

import numpy as np
import torch
import tqdm

import manager_folds
import model_event_unet
import config
import convert_to_npy_naive

FOLDER = "./inference_confidence_statistics/confidence_labels"

def inference(model_dir, validation_entries, all_data,
              hidden_blocks, hidden_channels, bottleneck_factor, squeeze_excitation, kernel_size,
              attention_blocks, attention_bottleneck, attention_mode, upconv_channels_override,
              expand, use_batch_norm, use_anglez_only, use_enmo_only,
              prediction_length, batch_size, use_time_information,

              stride_count=4, flip_augmentation=False,
              show_tqdm_bar=True):
    # init model
    in_channels = 1 if (use_anglez_only or use_enmo_only) else 2
    model = model_event_unet.EventConfidenceUnet(in_channels, hidden_channels, kernel_size=kernel_size,
                                                 blocks=hidden_blocks,
                                                 bottleneck_factor=bottleneck_factor,
                                                 squeeze_excitation=squeeze_excitation,
                                                 squeeze_excitation_bottleneck_factor=4,
                                                 dropout=0.0, dropout_pos_embeddings=False,
                                                 use_batch_norm=use_batch_norm, attn_out_channels=2,
                                                 attention_bottleneck=attention_bottleneck,
                                                 expected_attn_input_length=17280 + (2 * expand),
                                                 attention_blocks=attention_blocks,
                                                 upconv_channels_override=upconv_channels_override,
                                                 attention_mode=attention_mode,
                                                 use_time_input=use_time_information)
    model = model.to(config.device)
    model.eval()

    # load model
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))

    # inference
    all_preds = {}
    with torch.no_grad():
        if show_tqdm_bar:
            pbar = tqdm.tqdm(total=len(validation_entries))
        for k in range(len(validation_entries)):
            series_id = validation_entries[k]  # series ids

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

            preds = model_event_unet.event_confidence_inference(model=model, time_series=accel_data,
                                                                batch_size=batch_size,
                                                                prediction_length=prediction_length,
                                                                expand=expand, times=times,
                                                                stride_count=stride_count,
                                                                flip_augmentation=flip_augmentation)

            # save to dict
            all_preds[series_id] = {
                "onset": preds[0, :],
                "wakeup": preds[1, :]
            }

            if show_tqdm_bar:
                pbar.update(1)
        if show_tqdm_bar:
            pbar.close()

    return all_preds

if __name__ == "__main__":
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)

    parser = argparse.ArgumentParser("Inference script for models trained on regression events data")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[1, 6, 8, 23, 8], help="Number of hidden 2d blocks for ResNet backbone.")
    parser.add_argument("--hidden_channels", type=int, nargs="+", default=[2], help="Number of hidden channels. Default 2. Can be a list to specify num channels in each downsampled layer.")
    parser.add_argument("--bottleneck_factor", type=int, default=4, help="The bottleneck factor of the ResNet backbone. Default 4.")
    parser.add_argument("--squeeze_excitation", action="store_false", help="Whether to use squeeze and excitation. Default True.")
    parser.add_argument("--kernel_size", type=int, default=11, help="Kernel size for the first layer. Default 11.")
    parser.add_argument("--attention_blocks", type=int, default=4, help="Number of attention blocks to use. Default 4.")
    parser.add_argument("--attention_bottleneck", type=int, default=None, help="The bottleneck factor of the attention module. Default None.")
    parser.add_argument("--attention_mode", type=str, default="learned", help="Attention mode. Must be 'learned' or 'length'")
    parser.add_argument("--upconv_channels_override", type=int, default=None, help="Number of fixed channels for the upsampling path. Default None, do not override.")
    parser.add_argument("--expand", type=int, default=0, help="Expand the intervals by this amount. Default 0.")
    parser.add_argument("--use_batch_norm", action="store_true", help="Whether to use batch norm. Default False.")
    parser.add_argument("--prediction_length", type=int, default=17280, help="Number of timesteps to predict. Default 17280.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size. Default 512.")
    parser.add_argument("--use_time_information", action="store_true", help="Whether to use time information. Default False.")
    parser.add_argument("--IOU_intersection_width", type=int, default=30 * 12, help="Intersection width for IOU score. Default 30 * 12.")
    parser.add_argument("--IOU_union_width", type=int, default=60 * 12, help="Union width for IOU score. Default 60 * 12.")
    config.add_argparse_arguments(parser)
    args = parser.parse_args()

    # initialize gpu
    config.parse_args(args)

    # get data parameters
    hidden_blocks = args.hidden_blocks
    hidden_channels = args.hidden_channels
    bottleneck_factor = args.bottleneck_factor
    squeeze_excitation = args.squeeze_excitation
    kernel_size = args.kernel_size
    attention_blocks = args.attention_blocks
    attention_bottleneck = args.attention_bottleneck
    attention_mode = args.attention_mode
    upconv_channels_override = args.upconv_channels_override
    expand = args.expand
    use_batch_norm = args.use_batch_norm
    prediction_length = args.prediction_length
    batch_size = args.batch_size
    use_time_information = args.use_time_information
    IOU_intersection_width = args.IOU_intersection_width
    IOU_union_width = args.IOU_union_width

    args = {
        "hidden_blocks": hidden_blocks,
        "hidden_channels": hidden_channels,
        "bottleneck_factor": bottleneck_factor,
        "squeeze_excitation": squeeze_excitation,
        "kernel_size": kernel_size,
        "attention_blocks": attention_blocks,
        "attention_bottleneck": attention_bottleneck,
        "attention_mode": attention_mode,
        "upconv_channels_override": upconv_channels_override,
        "expand": expand,
        "use_batch_norm": use_batch_norm,
        "prediction_length": prediction_length,
        "batch_size": batch_size,
        "use_time_information": use_time_information,
        "stride_count": 4,
        "flip_augmentation": False
    }

    # load data
    all_data = convert_to_npy_naive.load_all_data_into_dict()

    # load options
    with open(os.path.join("./inference_confidence_statistics", "inference_confidence_preds_options.json"), "r") as f:
        options = json.load(f)

    iou_score_converter = model_event_unet.ProbasIOUScoreConverter(intersection_width=IOU_intersection_width, union_width=IOU_union_width, device=config.device)

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

            all_preds = inference(model_dir, validation_entries, all_data,
                                    opt_args["hidden_blocks"], opt_args["hidden_channels"], opt_args["bottleneck_factor"], opt_args["squeeze_excitation"], opt_args["kernel_size"],
                                    opt_args["attention_blocks"], opt_args["attention_bottleneck"], opt_args["attention_mode"], opt_args["upconv_channels_override"],
                                    opt_args["expand"], opt_args["use_batch_norm"], use_anglez_only, use_enmo_only,
                                    opt_args["prediction_length"], opt_args["batch_size"], opt_args["use_time_information"],

                                    stride_count=opt_args["stride_count"], flip_augmentation=opt_args["flip_augmentation"])

            for series_id in tqdm.tqdm(validation_entries):
                preds = all_preds[series_id]
                np.save(os.path.join(out_dir, "{}_onset.npy".format(series_id)), preds["onset"])
                np.save(os.path.join(out_dir, "{}_wakeup.npy".format(series_id)), preds["wakeup"])
                np.save(os.path.join(out_dir, "{}_IOU_onset.npy".format(series_id)), iou_score_converter.convert(preds["onset"]))
                np.save(os.path.join(out_dir, "{}_IOU_wakeup.npy".format(series_id)), iou_score_converter.convert(preds["wakeup"]))
