import os
import argparse
import shutil

import numpy as np
import torch
import tqdm

import manager_folds
import manager_models
import model_unet
import config
import convert_to_h5py_naive

if __name__ == "__main__":
    FOLDER = "pseudo_labels"
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)

    parser = argparse.ArgumentParser("Inference script for models trained on cleaned data")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[1, 6, 8, 23, 8],
                        help="Number of hidden 2d blocks for ResNet backbone.")
    parser.add_argument("--hidden_channels", type=int, default=32, help="Number of hidden channels. Default None.")
    parser.add_argument("--bottleneck_factor", type=int, default=4,
                        help="The bottleneck factor of the ResNet backbone. Default 4.")
    parser.add_argument("--squeeze_excitation", action="store_false",
                        help="Whether to use squeeze and excitation. Default True.")
    parser.add_argument("--disable_odd_random_shift", action="store_true",
                        help="Whether to disable odd random shift. Default False.")
    parser.add_argument("--use_batch_norm", action="store_true", help="Whether to use batch norm. Default False.")
    parser.add_argument("--data_to_infer", type=str, help="Which data to infer.", required=True)
    parser.add_argument("--load_model", type=str, help="Which model to load.", required=True)
    config.add_argparse_arguments(parser)

    # parse args
    args = parser.parse_args()
    config.parse_args(args)
    previous_model_path = os.path.join(manager_models.model_path, args.load_model)
    assert os.path.exists(previous_model_path), "Previous model checkpoint does not exist"
    data_to_infer = manager_folds.load_dataset(args.data_to_infer)

    hidden_blocks = args.hidden_blocks
    hidden_channels = args.hidden_channels
    bottleneck_factor = args.bottleneck_factor
    squeeze_excitation = args.squeeze_excitation
    disable_odd_random_shift = args.disable_odd_random_shift
    use_batch_norm = args.use_batch_norm

    # init model
    model = model_unet.Unet(2, hidden_channels, kernel_size=11, blocks=hidden_blocks,
                            bottleneck_factor=bottleneck_factor, squeeze_excitation=squeeze_excitation,
                            squeeze_excitation_bottleneck_factor=4,
                            odd_random_shift_training=(not disable_odd_random_shift),
                            dropout=0.0, use_batch_norm=use_batch_norm)
    model = model.to(config.device)
    model.eval()

    # load model
    model.load_state_dict(torch.load(os.path.join(previous_model_path, "model.pt")))

    # load data
    all_data = convert_to_h5py_naive.load_all_data_into_dict()

    # inference
    output_folder = os.path.join(FOLDER, args.load_model + "_" + args.data_to_infer)
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)
    with torch.no_grad():
        with tqdm.tqdm(total=len(data_to_infer)) as pbar:
            for k in range(len(data_to_infer)):
                series_id = data_to_infer[k]  # series ids

                # load the batch
                accel_data_batch = all_data[series_id]["accel"]
                accel_data_batch = torch.tensor(accel_data_batch, dtype=torch.float32, device=config.device).unsqueeze(0)

                # pad such that lengths is a multiple of 16
                pad_length = 16 - (accel_data_batch.shape[-1] % 16)
                if pad_length == 16:
                    pad_length = 0
                pad_left = pad_length // 2
                pad_right = pad_length - pad_left

                # get probas now
                accel_data_batch = torch.nn.functional.pad(accel_data_batch, (pad_left, pad_right), mode="replicate")
                pred_logits = model(accel_data_batch)  # shape (batch_size, 1, T), where batch_size = 1
                pred_logits = pred_logits[..., pad_left:(pred_logits.shape[-1] - pad_right)]
                pred_probas = torch.sigmoid(pred_logits).squeeze(0).squeeze(0).cpu().numpy()

                # save to folder
                np.save(os.path.join(output_folder, "{}.npy".format(series_id)), pred_probas)

                pbar.update(1)
