import os
import argparse
import shutil

import numpy as np
import torch
import tqdm

import manager_folds
import manager_models
import model_unet
import model_attention_unet
import config
import convert_to_h5py_naive

FOLDER = "regression_labels"

if __name__ == "__main__":
    FOLDERS = [
        os.path.join(FOLDER, "large_onset"),
        os.path.join(FOLDER, "mid_onset"),
        os.path.join(FOLDER, "small_onset"),
        os.path.join(FOLDER, "large_wakeup"),
        os.path.join(FOLDER, "mid_wakeup"),
        os.path.join(FOLDER, "small_wakeup"),
    ]
    FOLDERS_DICT = {
        "large_onset": FOLDERS[0],
        "mid_onset": FOLDERS[1],
        "small_onset": FOLDERS[2],
        "large_wakeup": FOLDERS[3],
        "mid_wakeup": FOLDERS[4],
        "small_wakeup": FOLDERS[5],
    }
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)
    for folder in FOLDERS:
        if not os.path.isdir(folder):
            os.mkdir(folder)

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

    # init model
    in_channels = 1 if (use_anglez_only or use_enmo_only) else 2
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
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))

    # load data
    all_data = convert_to_h5py_naive.load_all_data_into_dict()
    target_multiple = 3 * (2 ** (len(hidden_blocks) - 2))

    # inference
    with torch.no_grad():
        with tqdm.tqdm(total=len(validation_entries)) as pbar:
            for k in range(len(validation_entries)):
                series_id = validation_entries[k]  # series ids

                # load the batch
                accel_data = all_data[series_id]["accel"]
                series_length = accel_data.shape[1]
                target_length = (series_length // target_multiple) * target_multiple
                start = (series_length - target_length) // 2
                end = start + target_length
                end_contraction = series_length - end
                accel_data = accel_data[:, start:end]

                accel_data_batch = torch.tensor(accel_data, dtype=torch.float32, device=config.device).unsqueeze(0)
                if use_anglez_only:
                    accel_data_batch = accel_data_batch[:, 0:1, :]
                elif use_enmo_only:
                    accel_data_batch = accel_data_batch[:, 1:2, :]

                # get predictions now
                pred_small, pred_mid, pred_large = model(accel_data_batch, ret_type="deep")
                #pred_small, pred_mid, pred_large = torch.sigmoid(pred_small), torch.sigmoid(pred_mid), torch.sigmoid(pred_large)
                pred_small, pred_mid, pred_large = pred_small.squeeze(0), pred_mid.squeeze(0), pred_large.squeeze(0)
                pred_small, pred_mid, pred_large = pred_small.cpu().numpy(), pred_mid.cpu().numpy(), pred_large.cpu().numpy()
                pred_small, pred_mid, pred_large = np.pad(pred_small, ((0, 0), (start, end_contraction)), mode="constant"), \
                                                     np.pad(pred_mid, ((0, 0), (start, end_contraction)), mode="constant"), \
                                                        np.pad(pred_large, ((0, 0), (start, end_contraction)), mode="constant")

                # save to folder
                np.save(os.path.join(FOLDERS_DICT["small_onset"], "{}.npy".format(series_id)), pred_small[0, :])
                np.save(os.path.join(FOLDERS_DICT["mid_onset"], "{}.npy".format(series_id)), pred_mid[0, :])
                np.save(os.path.join(FOLDERS_DICT["large_onset"], "{}.npy".format(series_id)), pred_large[0, :])
                np.save(os.path.join(FOLDERS_DICT["small_wakeup"], "{}.npy".format(series_id)), pred_small[1, :])
                np.save(os.path.join(FOLDERS_DICT["mid_wakeup"], "{}.npy".format(series_id)), pred_mid[1, :])
                np.save(os.path.join(FOLDERS_DICT["large_wakeup"], "{}.npy".format(series_id)), pred_large[1, :])

                pbar.update(1)
