import os
import argparse
import json

import numpy as np
import torch
import tqdm

import manager_folds
import model_event_unet
import kernel_utils
import config
import convert_to_npy_naive

FOLDER = "./inference_regression_statistics/regression_labels"

def inference(model_dir, out_folder, validation_entries, target_multiple, use_sigmas):
    # specify folders, and create them if they don't exist
    FOLDERS_DICT = {
        "regression": os.path.join(out_folder, "regression")
    }
    ker_vals = [2, 6, 9, 12, 36, 90, 360]
    if use_sigmas:
        FOLDERS_DICT["gaussian_kernel"] = os.path.join(out_folder, "gaussian_kernel")
        FOLDERS_DICT["laplace_kernel"] = os.path.join(out_folder, "laplace_kernel")
        FOLDERS_DICT["huber_kernel"] = os.path.join(out_folder, "huber_kernel")
    else:
        for k in ker_vals:
            FOLDERS_DICT["gaussian_kernel{}".format(k)] = os.path.join(out_folder, "gaussian_kernel{}".format(k))
            FOLDERS_DICT["laplace_kernel{}".format(k)] = os.path.join(out_folder, "laplace_kernel{}".format(k))
            FOLDERS_DICT["huber_kernel{}".format(k)] = os.path.join(out_folder, "huber_kernel{}".format(k))

    for key in FOLDERS_DICT:
        if not os.path.isdir(FOLDERS_DICT[key]):
            os.mkdir(FOLDERS_DICT[key])

    # load model
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))
    model.eval()

    # inference
    with torch.no_grad():
        with tqdm.tqdm(total=len(validation_entries)) as pbar:
            for k in range(len(validation_entries)):
                series_id = validation_entries[k]  # series ids

                # load the batch
                accel_data = all_data[series_id]["accel"]
                if use_anglez_only:
                    accel_data = accel_data[0:1, :]
                elif use_enmo_only:
                    accel_data = accel_data[1:2, :]

                # get predictions now
                if os.path.isfile(os.path.join(FOLDERS_DICT["regression"], "{}_onset.npy".format(series_id))):
                    if use_sigmas:
                        preds = torch.tensor(np.stack([
                            np.load(os.path.join(FOLDERS_DICT["regression"], "{}_onset.npy".format(series_id))),
                            np.load(os.path.join(FOLDERS_DICT["regression"], "{}_wakeup.npy".format(series_id))),
                            np.load(os.path.join(FOLDERS_DICT["regression"], "{}_onset_sigmas.npy".format(series_id))),
                            np.load(os.path.join(FOLDERS_DICT["regression"], "{}_wakeup_sigmas.npy".format(series_id)))
                        ], axis=0), dtype=torch.float32, device=config.device)
                    else:
                        preds = torch.tensor(np.stack([
                            np.load(os.path.join(FOLDERS_DICT["regression"], "{}_onset.npy".format(series_id))),
                            np.load(os.path.join(FOLDERS_DICT["regression"], "{}_wakeup.npy".format(series_id)))
                        ], axis=0), dtype=torch.float32, device=config.device)
                else:
                    preds = model_event_unet.event_regression_inference(model, accel_data, target_multiple=target_multiple, return_torch_tensor=True)

                    # save to folder
                    np.save(os.path.join(FOLDERS_DICT["regression"], "{}_onset.npy".format(series_id)), preds[0, :].cpu().numpy())
                    np.save(os.path.join(FOLDERS_DICT["regression"], "{}_wakeup.npy".format(series_id)), preds[1, :].cpu().numpy())
                    if use_sigmas:
                        np.save(os.path.join(FOLDERS_DICT["regression"], "{}_onset_sigmas.npy".format(series_id)), preds[2, :].cpu().numpy())
                        np.save(os.path.join(FOLDERS_DICT["regression"], "{}_wakeup_sigmas.npy".format(series_id)), preds[3, :].cpu().numpy())

                # generate kernel predictions
                if use_sigmas:
                    # onset
                    npy_file = os.path.join(FOLDERS_DICT["gaussian_kernel"], "{}_onset.npy".format(series_id))
                    if os.path.isfile(npy_file):
                        onset_kernel_pred = np.load(npy_file)
                    else:
                        onset_kernel_pred = kernel_utils.generate_kernel_preds_sigma_gpu(preds[0, :], sigmas_array=preds[2, :], device=preds.device,
                                                                                         kernel_generating_function=kernel_utils.generate_kernel_preds_sigmas)
                        np.save(npy_file, onset_kernel_pred)
                    local_maximums = (onset_kernel_pred[1:-1] > onset_kernel_pred[0:-2]) & (onset_kernel_pred[1:-1] > onset_kernel_pred[2:])
                    local_maximums = np.argwhere(local_maximums).flatten() + 1
                    local_maximums = np.stack([
                        local_maximums.astype(np.float32),
                        onset_kernel_pred[local_maximums].astype(np.float32)
                    ], axis=0)
                    np.save(os.path.join(FOLDERS_DICT["gaussian_kernel"], "{}_onset_locmax.npy".format(series_id)), local_maximums)
                    
                    npy_file = os.path.join(FOLDERS_DICT["laplace_kernel"], "{}_onset.npy".format(series_id))
                    if os.path.isfile(npy_file):
                        onset_kernel_pred = np.load(npy_file)
                    else:
                        onset_kernel_pred = kernel_utils.generate_kernel_preds_sigma_gpu(preds[0, :], sigmas_array=preds[2, :], device=preds.device,
                                                                                         kernel_generating_function=kernel_utils.generate_kernel_preds_laplace_sigmas)
                        np.save(npy_file, onset_kernel_pred)
                    local_maximums = (onset_kernel_pred[1:-1] > onset_kernel_pred[0:-2]) & (onset_kernel_pred[1:-1] > onset_kernel_pred[2:])
                    local_maximums = np.argwhere(local_maximums).flatten() + 1
                    local_maximums = np.stack([
                        local_maximums.astype(np.float32),
                        onset_kernel_pred[local_maximums].astype(np.float32)
                    ], axis=0)
                    np.save(os.path.join(FOLDERS_DICT["laplace_kernel"], "{}_onset_locmax.npy".format(series_id)), local_maximums)
                    
                    npy_file = os.path.join(FOLDERS_DICT["huber_kernel"], "{}_onset.npy".format(series_id))
                    if os.path.isfile(npy_file):
                        onset_kernel_pred = np.load(npy_file)
                    else:
                        onset_kernel_pred = kernel_utils.generate_kernel_preds_sigma_gpu(preds[0, :], sigmas_array=preds[2, :], device=preds.device,
                                                                                         kernel_generating_function=kernel_utils.generate_kernel_preds_huber_sigmas)
                        np.save(npy_file, onset_kernel_pred)
                    local_maximums = (onset_kernel_pred[1:-1] > onset_kernel_pred[0:-2]) & (onset_kernel_pred[1:-1] > onset_kernel_pred[2:])
                    local_maximums = np.argwhere(local_maximums).flatten() + 1
                    local_maximums = np.stack([
                        local_maximums.astype(np.float32),
                        onset_kernel_pred[local_maximums].astype(np.float32)
                    ], axis=0)
                    np.save(os.path.join(FOLDERS_DICT["huber_kernel"], "{}_onset_locmax.npy".format(series_id)), local_maximums)

                    # wakeup
                    npy_file = os.path.join(FOLDERS_DICT["gaussian_kernel"], "{}_wakeup.npy".format(series_id))
                    if os.path.isfile(npy_file):
                        wakeup_kernel_pred = np.load(npy_file)
                    else:
                        wakeup_kernel_pred = kernel_utils.generate_kernel_preds_sigma_gpu(preds[1, :], sigmas_array=preds[3, :], device=preds.device,
                                                                                          kernel_generating_function=kernel_utils.generate_kernel_preds_sigmas)
                        np.save(npy_file, wakeup_kernel_pred)
                    local_maximums = (wakeup_kernel_pred[1:-1] > wakeup_kernel_pred[0:-2]) & (wakeup_kernel_pred[1:-1] > wakeup_kernel_pred[2:])
                    local_maximums = np.argwhere(local_maximums).flatten() + 1
                    local_maximums = np.stack([
                        local_maximums.astype(np.float32),
                        wakeup_kernel_pred[local_maximums].astype(np.float32)
                    ], axis=0)
                    np.save(os.path.join(FOLDERS_DICT["gaussian_kernel"], "{}_wakeup_locmax.npy".format(series_id)), local_maximums)

                    npy_file = os.path.join(FOLDERS_DICT["laplace_kernel"], "{}_wakeup.npy".format(series_id))
                    if os.path.isfile(npy_file):
                        wakeup_kernel_pred = np.load(npy_file)
                    else:
                        wakeup_kernel_pred = kernel_utils.generate_kernel_preds_sigma_gpu(preds[1, :], sigmas_array=preds[3, :], device=preds.device,
                                                                                          kernel_generating_function=kernel_utils.generate_kernel_preds_laplace_sigmas)
                        np.save(npy_file, wakeup_kernel_pred)
                    local_maximums = (wakeup_kernel_pred[1:-1] > wakeup_kernel_pred[0:-2]) & (wakeup_kernel_pred[1:-1] > wakeup_kernel_pred[2:])
                    local_maximums = np.argwhere(local_maximums).flatten() + 1
                    local_maximums = np.stack([
                        local_maximums.astype(np.float32),
                        wakeup_kernel_pred[local_maximums].astype(np.float32)
                    ], axis=0)
                    np.save(os.path.join(FOLDERS_DICT["laplace_kernel"], "{}_wakeup_locmax.npy".format(series_id)), local_maximums)

                    npy_file = os.path.join(FOLDERS_DICT["huber_kernel"], "{}_wakeup.npy".format(series_id))
                    if os.path.isfile(npy_file):
                        wakeup_kernel_pred = np.load(npy_file)
                    else:
                        wakeup_kernel_pred = kernel_utils.generate_kernel_preds_sigma_gpu(preds[1, :], sigmas_array=preds[3, :], device=preds.device,
                                                                                          kernel_generating_function=kernel_utils.generate_kernel_preds_huber_sigmas)
                        np.save(npy_file, wakeup_kernel_pred)
                    local_maximums = (wakeup_kernel_pred[1:-1] > wakeup_kernel_pred[0:-2]) & (wakeup_kernel_pred[1:-1] > wakeup_kernel_pred[2:])
                    local_maximums = np.argwhere(local_maximums).flatten() + 1
                    local_maximums = np.stack([
                        local_maximums.astype(np.float32),
                        wakeup_kernel_pred[local_maximums].astype(np.float32)
                    ], axis=0)
                    np.save(os.path.join(FOLDERS_DICT["huber_kernel"], "{}_wakeup_locmax.npy".format(series_id)), local_maximums)
                else:
                    for ker_val in ker_vals:
                        npy_file = os.path.join(FOLDERS_DICT["gaussian_kernel{}".format(ker_val)], "{}_onset.npy".format(series_id))
                        if os.path.isfile(npy_file):
                            onset_gaussian_pred = np.load(npy_file)
                        else:
                            onset_gaussian_pred = kernel_utils.generate_kernel_preds_gpu(preds[0, :], device=preds.device,
                                                                                         kernel_generating_function=kernel_utils.generate_kernel_preds,
                                                                                         kernel_radius=ker_val, max_clip=240 + 5 * ker_val)
                            np.save(npy_file, onset_gaussian_pred)
                        local_maximums = (onset_gaussian_pred[1:-1] > onset_gaussian_pred[0:-2]) & (onset_gaussian_pred[1:-1] > onset_gaussian_pred[2:])
                        local_maximums = np.argwhere(local_maximums).flatten() + 1
                        local_maximums = np.stack([
                            local_maximums.astype(np.float32),
                            onset_gaussian_pred[local_maximums].astype(np.float32)
                        ], axis=0)
                        np.save(os.path.join(FOLDERS_DICT["gaussian_kernel{}".format(ker_val)], "{}_onset_locmax.npy".format(series_id)), local_maximums)

                    for ker_val in ker_vals:
                        npy_file = os.path.join(FOLDERS_DICT["laplace_kernel{}".format(ker_val)], "{}_onset.npy".format(series_id))
                        if os.path.isfile(npy_file):
                            onset_laplace_pred = np.load(npy_file)
                        else:
                            onset_laplace_pred = kernel_utils.generate_kernel_preds_gpu(preds[0, :], device=preds.device,
                                                                                         kernel_generating_function=kernel_utils.generate_kernel_preds_laplace,
                                                                                         kernel_radius=ker_val, max_clip=240 + 5 * ker_val)
                            np.save(npy_file, onset_laplace_pred)
                        local_maximums = (onset_laplace_pred[1:-1] > onset_laplace_pred[0:-2]) & (onset_laplace_pred[1:-1] > onset_laplace_pred[2:])
                        local_maximums = np.argwhere(local_maximums).flatten() + 1
                        local_maximums = np.stack([
                            local_maximums.astype(np.float32),
                            onset_laplace_pred[local_maximums].astype(np.float32)
                        ], axis=0)
                        np.save(os.path.join(FOLDERS_DICT["laplace_kernel{}".format(ker_val)], "{}_onset_locmax.npy".format(series_id)), local_maximums)

                    for ker_val in ker_vals:
                        npy_file = os.path.join(FOLDERS_DICT["huber_kernel{}".format(ker_val)], "{}_onset.npy".format(series_id))
                        if os.path.isfile(npy_file):
                            onset_huber_pred = np.load(npy_file)
                        else:
                            onset_huber_pred = kernel_utils.generate_kernel_preds_gpu(preds[0, :], device=preds.device,
                                                                                         kernel_generating_function=kernel_utils.generate_kernel_preds_huber,
                                                                                         kernel_radius=ker_val, max_clip=240 + 5 * ker_val)
                            np.save(npy_file, onset_huber_pred)
                        local_maximums = (onset_huber_pred[1:-1] > onset_huber_pred[0:-2]) & (onset_huber_pred[1:-1] > onset_huber_pred[2:])
                        local_maximums = np.argwhere(local_maximums).flatten() + 1
                        local_maximums = np.stack([
                            local_maximums.astype(np.float32),
                            onset_huber_pred[local_maximums].astype(np.float32)
                        ], axis=0)
                        np.save(os.path.join(FOLDERS_DICT["huber_kernel{}".format(ker_val)], "{}_onset_locmax.npy".format(series_id)), local_maximums)

                    for ker_val in ker_vals:
                        npy_file = os.path.join(FOLDERS_DICT["gaussian_kernel{}".format(ker_val)], "{}_wakeup.npy".format(series_id))
                        if os.path.isfile(npy_file):
                            wakeup_gaussian_pred = np.load(npy_file)
                        else:
                            wakeup_gaussian_pred = kernel_utils.generate_kernel_preds_gpu(preds[1, :], device=preds.device,
                                                                                         kernel_generating_function=kernel_utils.generate_kernel_preds,
                                                                                         kernel_radius=ker_val, max_clip=240 + 5 * ker_val)
                            np.save(npy_file, wakeup_gaussian_pred)
                        local_maximums = (wakeup_gaussian_pred[1:-1] > wakeup_gaussian_pred[0:-2]) & (wakeup_gaussian_pred[1:-1] > wakeup_gaussian_pred[2:])
                        local_maximums = np.argwhere(local_maximums).flatten() + 1
                        local_maximums = np.stack([
                            local_maximums.astype(np.float32),
                            wakeup_gaussian_pred[local_maximums].astype(np.float32)
                        ], axis=0)
                        np.save(os.path.join(FOLDERS_DICT["gaussian_kernel{}".format(ker_val)], "{}_wakeup_locmax.npy".format(series_id)), local_maximums)

                    for ker_val in ker_vals:
                        npy_file = os.path.join(FOLDERS_DICT["laplace_kernel{}".format(ker_val)], "{}_wakeup.npy".format(series_id))
                        if os.path.isfile(npy_file):
                            wakeup_laplace_pred = np.load(npy_file)
                        else:
                            wakeup_laplace_pred = kernel_utils.generate_kernel_preds_gpu(preds[1, :], device=preds.device,
                                                                                         kernel_generating_function=kernel_utils.generate_kernel_preds_laplace,
                                                                                         kernel_radius=ker_val, max_clip=240 + 5 * ker_val)
                            np.save(npy_file, wakeup_laplace_pred)
                        local_maximums = (wakeup_laplace_pred[1:-1] > wakeup_laplace_pred[0:-2]) & (wakeup_laplace_pred[1:-1] > wakeup_laplace_pred[2:])
                        local_maximums = np.argwhere(local_maximums).flatten() + 1
                        local_maximums = np.stack([
                            local_maximums.astype(np.float32),
                            wakeup_laplace_pred[local_maximums].astype(np.float32)
                        ], axis=0)
                        np.save(os.path.join(FOLDERS_DICT["laplace_kernel{}".format(ker_val)], "{}_wakeup_locmax.npy".format(series_id)), local_maximums)

                    for ker_val in ker_vals:
                        npy_file = os.path.join(FOLDERS_DICT["huber_kernel{}".format(ker_val)], "{}_wakeup.npy".format(series_id))
                        if os.path.isfile(npy_file):
                            wakeup_huber_pred = np.load(npy_file)
                        else:
                            wakeup_huber_pred = kernel_utils.generate_kernel_preds_gpu(preds[1, :], device=preds.device,
                                                                                         kernel_generating_function=kernel_utils.generate_kernel_preds_huber,
                                                                                         kernel_radius=ker_val, max_clip=240 + 5 * ker_val)
                            np.save(npy_file, wakeup_huber_pred)
                        local_maximums = (wakeup_huber_pred[1:-1] > wakeup_huber_pred[0:-2]) & (wakeup_huber_pred[1:-1] > wakeup_huber_pred[2:])
                        local_maximums = np.argwhere(local_maximums).flatten() + 1
                        local_maximums = np.stack([
                            local_maximums.astype(np.float32),
                            wakeup_huber_pred[local_maximums].astype(np.float32)
                        ], axis=0)
                        np.save(os.path.join(FOLDERS_DICT["huber_kernel{}".format(ker_val)], "{}_wakeup_locmax.npy".format(series_id)), local_maximums)

                pbar.update(1)

if __name__ == "__main__":
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)

    parser = argparse.ArgumentParser("Inference script for models trained on regression events data")
    parser.add_argument("--use_anglez_only", action="store_true", help="Whether to use only anglez. Default False.")
    parser.add_argument("--use_enmo_only", action="store_true", help="Whether to use only enmo. Default False.")
    config.add_argparse_arguments(parser)
    args = parser.parse_args()

    # initialize gpu
    config.parse_args(args)

    # get data parameters
    use_anglez_only = args.use_anglez_only
    use_enmo_only = args.use_enmo_only

    assert not (use_anglez_only and use_enmo_only), "Cannot use both anglez only and enmo only."

    # this is constant
    blocks_length = 5

    # load data
    all_data = convert_to_npy_naive.load_all_data_into_dict()
    target_multiple = 3 * (2 ** (blocks_length - 2))

    # load options for statistics computation
    with open("./inference_regression_statistics/inference_regression_preds_options.json", "r") as f:
        options = json.load(f)

    for option in options:
        name = option["name"]
        models = option["models"]
        entries = option["entries"]
        use_sigmas = ("use_sigmas" in option) and option["use_sigmas"]
        print("Running inference on {}".format(name))

        # init model
        model = model_event_unet.EventRegressorUnet(use_learnable_sigma=use_sigmas)
        model = model.to(config.device)

        assert len(models) == len(entries), "Number of models and entries must be the same."
        assert all([os.path.isdir(os.path.join("models", model_name)) for model_name in models]), "All models must exist."

        name_formatted = name.replace(" ", "_").replace("(", "").replace(")", "")
        out_folder = os.path.join(FOLDER, name_formatted)
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)
            with open(os.path.join(out_folder, "name.txt"), "w") as f:
                f.write(name)
        for k in range(len(models)):
            model_name = models[k]
            entry = entries[k]

            model_dir = os.path.join("models", model_name)
            validation_entries = manager_folds.load_dataset(entry)

            inference(model_dir, out_folder, validation_entries, target_multiple, use_sigmas)