import os
import argparse
import shutil

import numpy as np
import torch
import tqdm

import manager_folds
import model_event_unet
import config
import convert_to_h5py_naive

FOLDER = "regression_labels"

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

    # get data parameters
    use_anglez_only = args.use_anglez_only
    use_enmo_only = args.use_enmo_only

    assert not (use_anglez_only and use_enmo_only), "Cannot use both anglez only and enmo only."

    # init model
    blocks_length = 5
    model = model_event_unet.EventRegressorUnet()
    model = model.to(config.device)
    model.eval()

    # load model
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))

    # load data
    all_data = convert_to_h5py_naive.load_all_data_into_dict()
    target_multiple = 3 * (2 ** (blocks_length - 2))

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
                preds = model(accel_data_batch, ret_type="deep")
                preds = preds.squeeze(0)
                preds = preds.cpu().numpy()
                preds = np.pad(preds, ((0, 0), (start, end_contraction)), mode="constant")

                # save to folder
                np.save(os.path.join(FOLDERS_DICT["onset"], "{}.npy".format(series_id)), preds[0, :])
                np.save(os.path.join(FOLDERS_DICT["wakeup"], "{}.npy".format(series_id)), preds[1, :])

                pbar.update(1)
