import os
import argparse
import shutil

import numpy as np
import torch
import tqdm

import manager_folds
import model_event_unet
import config
import convert_to_npy_naive

FOLDER = "confidence_labels"

def inference(model_dir, validation_entries, all_data,
              hidden_blocks, hidden_channels, bottleneck_factor, squeeze_excitation, kernel_size,
              attention_blocks, attention_bottleneck, upconv_channels_override,
              expand, use_batch_norm, use_anglez_only, use_enmo_only,
              prediction_length, batch_size,
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
                                                 upconv_channels_override=upconv_channels_override)
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

            preds = model_event_unet.event_confidence_inference(model=model, time_series=accel_data,
                                                                batch_size=batch_size,
                                                                prediction_length=prediction_length,
                                                                expand=expand)

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
    FOLDERS = [
        os.path.join(FOLDER, "onset"),
        os.path.join(FOLDER, "wakeup")
    ]
    FOLDERS_DICT = {
        "onset": FOLDERS[0],
        "wakeup": FOLDERS[1]
    }
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)
    for folder in FOLDERS:
        if not os.path.isdir(folder):
            os.mkdir(folder)

    parser = argparse.ArgumentParser("Inference script for models trained on regression events data")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[1, 6, 8, 23, 8], help="Number of hidden 2d blocks for ResNet backbone.")
    parser.add_argument("--hidden_channels", type=int, nargs="+", default=[2], help="Number of hidden channels. Default 2. Can be a list to specify num channels in each downsampled layer.")
    parser.add_argument("--bottleneck_factor", type=int, default=4, help="The bottleneck factor of the ResNet backbone. Default 4.")
    parser.add_argument("--squeeze_excitation", action="store_false", help="Whether to use squeeze and excitation. Default True.")
    parser.add_argument("--kernel_size", type=int, default=11, help="Kernel size for the first layer. Default 11.")
    parser.add_argument("--attention_blocks", type=int, default=4, help="Number of attention blocks to use. Default 4.")
    parser.add_argument("--attention_bottleneck", type=int, default=None, help="The bottleneck factor of the attention module. Default None.")
    parser.add_argument("--upconv_channels_override", type=int, default=None, help="Number of fixed channels for the upsampling path. Default None, do not override.")
    parser.add_argument("--expand", type=int, default=0, help="Expand the intervals by this amount. Default 0.")
    parser.add_argument("--use_batch_norm", action="store_true", help="Whether to use batch norm. Default False.")
    parser.add_argument("--use_anglez_only", action="store_true", help="Whether to use only anglez. Default False.")
    parser.add_argument("--use_enmo_only", action="store_true", help="Whether to use only enmo. Default False.")
    parser.add_argument("--prediction_length", type=int, default=17280, help="Number of timesteps to predict. Default 17280.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size. Default 512.")
    parser.add_argument("--load_model", type=str, required=True, help="The model to load.")
    manager_folds.add_argparse_arguments(parser)
    config.add_argparse_arguments(parser)
    args = parser.parse_args()

    # check entries
    training_entries, validation_entries, train_dset_name, val_dset_name = manager_folds.parse_args(args)
    assert type(training_entries) == list
    assert type(validation_entries) == list
    print("Training dataset: {}".format(train_dset_name))
    print("Validation dataset: {}".format(val_dset_name))

    # initialize gpu
    config.parse_args(args)

    # get model directories
    model_dir = os.path.join("models", args.load_model)
    assert os.path.isdir(model_dir), "Model directory does not exist."

    # get data parameters
    hidden_blocks = args.hidden_blocks
    hidden_channels = args.hidden_channels
    bottleneck_factor = args.bottleneck_factor
    squeeze_excitation = args.squeeze_excitation
    kernel_size = args.kernel_size
    attention_blocks = args.attention_blocks
    attention_bottleneck = args.attention_bottleneck
    upconv_channels_override = args.upconv_channels_override
    expand = args.expand
    use_batch_norm = args.use_batch_norm
    use_anglez_only = args.use_anglez_only
    use_enmo_only = args.use_enmo_only
    prediction_length = args.prediction_length
    batch_size = args.batch_size

    assert not (use_anglez_only and use_enmo_only), "Cannot use both anglez only and enmo only."

    # load data
    all_data = convert_to_npy_naive.load_all_data_into_dict()

    # inference
    all_preds = inference(model_dir, validation_entries, all_data,
                            hidden_blocks, hidden_channels, bottleneck_factor, squeeze_excitation, kernel_size,
                            attention_blocks, attention_bottleneck, upconv_channels_override,
                            expand, use_batch_norm, use_anglez_only, use_enmo_only,
                            prediction_length, batch_size)

    for series_id in tqdm.tqdm(validation_entries):
        preds = all_preds[series_id]
        np.save(os.path.join(FOLDERS_DICT["onset"], "{}.npy".format(series_id)), preds["onset"])
        np.save(os.path.join(FOLDERS_DICT["wakeup"], "{}.npy".format(series_id)), preds["wakeup"])
