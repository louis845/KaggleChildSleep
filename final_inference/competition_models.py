import json
import os
import time

import torch
import numpy as np
import numba
import numba.cuda
import stumpy

import postprocessing
import model_event_density_unet
import kernel_utils

class CompetitionModels:
    def __init__(self, model_config_file, models_root_dir: str, device: torch.device):
        with open(model_config_file, "r") as f:
            self.model_config = json.load(f)
        self.models_root_dir = models_root_dir
        self.device = device

        self.regression_models = []
        self.confidence_models = []
        self.stumpy_device = None

        self.regression_cutoff = None
        self.regression_pruning = None
        self.confidence_cutoff = None
        self.confidence_aug_cutoff = None
        self.matrix_profile_downsampling_rate = None

    def load_regression_models(self):
        for cfg in self.model_config["regression_models"]:
            regression_cfg = {
                "hidden_blocks": [2, 2, 2, 2, 3],
                "hidden_channels": [4, 4, 8, 16, 32],
                "pred_width": 120,
                "kernel_size": 9,
                "use_sigmas": False,
                "use_swa": False
            }
            if "hidden_blocks" in cfg:
                regression_cfg["hidden_blocks"] = cfg["hidden_blocks"]
            if "hidden_channels" in cfg:
                regression_cfg["hidden_channels"] = cfg["hidden_channels"]
            if "pred_width" in cfg:
                regression_cfg["pred_width"] = cfg["pred_width"]
            if "use_sigmas" in cfg:
                regression_cfg["use_sigmas"] = cfg["use_sigmas"]
            if "use_swa" in cfg:
                regression_cfg["use_swa"] = cfg["use_swa"]
            regression_cfg["model_name"] = cfg["model_name"]

            blocks_length = len(regression_cfg["hidden_blocks"])
            target_multiple = 3 * (2 ** (blocks_length - 2))

            model = model_event_density_unet.EventRegressorUnet(use_learnable_sigma=regression_cfg["use_sigmas"],
                                                        blocks=regression_cfg["hidden_blocks"],
                                                        hidden_channels=regression_cfg["hidden_channels"])
            model.to(self.device)
            if regression_cfg["use_swa"]:
                model = torch.optim.swa_utils.AveragedModel(model)
            model.load_state_dict(torch.load(os.path.join(self.models_root_dir,
                                                          "{}.pt".format(regression_cfg["model_name"])
                                                          ), weights_only=True, map_location=self.device))
            model.eval()

            model_pkg = {
                "model": model,
                "model_name": regression_cfg["model_name"],
                "pred_width": regression_cfg["pred_width"],
                "kernel_size": regression_cfg["kernel_size"],
                "use_sigmas": regression_cfg["use_sigmas"],
                "use_swa": regression_cfg["use_swa"],
                "target_multiple": target_multiple
            }

            self.regression_models.append(model_pkg)

    def load_confidence_models(self):
        for cfg in self.model_config["confidence_models"]:
            confidence_cfg = {
                "attention_blocks": 3,
                "attention_mode": "length",
                "stride_count": 4,
                "use_time_information": False,
                "input_data": "anglez",
                "training_strategy": "density_only",
                "input_length": 17280,
                "expand": 8640,
                "use_swa": False
            }

            if "attention_blocks" in cfg:
                confidence_cfg["attention_blocks"] = cfg["attention_blocks"]
            if "attention_mode" in cfg:
                confidence_cfg["attention_mode"] = cfg["attention_mode"]
            if "stride_count" in cfg:
                confidence_cfg["stride_count"] = cfg["stride_count"]
            if "use_time_information" in cfg:
                confidence_cfg["use_time_information"] = cfg["use_time_information"]
            if "input_data" in cfg:
                confidence_cfg["input_data"] = cfg["input_data"]
            if "training_strategy" in cfg:
                confidence_cfg["training_strategy"] = cfg["training_strategy"]
            if "input_length" in cfg:
                confidence_cfg["input_length"] = cfg["input_length"]
            if "expand" in cfg:
                confidence_cfg["expand"] = cfg["expand"]
            if "use_swa" in cfg:
                confidence_cfg["use_swa"] = cfg["use_swa"]
            confidence_cfg["model_name"] = cfg["model_name"]
            print(confidence_cfg["model_name"])

            assert confidence_cfg["input_data"] in ["anglez", "enmo", "both"]

            model = model_event_density_unet.EventDensityUnet(in_channels=1,
                                                         attention_blocks=confidence_cfg["attention_blocks"],
                                                         attention_mode=confidence_cfg["attention_mode"],
                                                         use_time_input=confidence_cfg["use_time_information"],
                                                         training_strategy=confidence_cfg["training_strategy"],
                                                         input_interval_length=confidence_cfg["input_length"],
                                                         input_expand_radius=confidence_cfg["expand"])
            model.to(self.device)
            if confidence_cfg["use_swa"]:
                model = torch.optim.swa_utils.AveragedModel(model)
            model.load_state_dict(torch.load(os.path.join(self.models_root_dir,
                                                          "{}.pt".format(confidence_cfg["model_name"])
                                                          ), weights_only=True, map_location=self.device))
            model.eval()

            model_pkg = {
                "model": model,
                "model_name": confidence_cfg["model_name"],
                "prediction_length": confidence_cfg["input_length"],
                "expand": confidence_cfg["expand"],
                "use_time_information": confidence_cfg["use_time_information"],
                "input_data": confidence_cfg["input_data"],
                "use_swa": confidence_cfg["use_swa"],
                "stride_count": confidence_cfg["stride_count"]
            }

            self.confidence_models.append(model_pkg)

    def load_models(self):
        self.load_regression_models()
        self.load_confidence_models()

    def set_parameters(self, regression_cutoff, regression_pruning, confidence_cutoff, confidence_aug_cutoff, matrix_profile_downsampling_rate):
        self.regression_cutoff = regression_cutoff
        self.regression_pruning = regression_pruning
        self.confidence_cutoff = confidence_cutoff
        self.confidence_aug_cutoff = confidence_aug_cutoff
        self.matrix_profile_downsampling_rate = matrix_profile_downsampling_rate

    def run_inference(self, series_id: str, accel_data_anglez: np.ndarray, accel_data_enmo: np.ndarray,
                      secs_corr: np.ndarray, mins_corr: np.ndarray, hours_corr: np.ndarray,
                      models_subset: list = None, use_matrix_profile_pruning: bool = False):
        ## cfg values
        regression_cutoff = self.regression_cutoff
        regression_pruning = self.regression_pruning
        confidence_cutoff = self.confidence_cutoff
        confidence_aug_cutoff = self.confidence_aug_cutoff
        downsampling_rate = self.matrix_profile_downsampling_rate
        batch_size = 512

        ## find regression locations first
        ctime = time.time()
        onset_kernel_values = None
        wakeup_kernel_values = None
        num_regression_models = 0
        for regression_model_pkg in self.regression_models:
            # ensemble the kernel values
            model = regression_model_pkg["model"]
            model_name = regression_model_pkg["model_name"]
            pred_width = regression_model_pkg["pred_width"]
            kernel_size = regression_model_pkg["kernel_size"]
            use_sigmas = regression_model_pkg["use_sigmas"]
            target_multiple = regression_model_pkg["target_multiple"]

            if models_subset is not None and model_name not in models_subset:
                continue
            num_regression_models += 1

            with torch.no_grad():
                preds_raw = model_event_density_unet.event_regression_inference(model, accel_data_anglez, target_multiple=target_multiple, return_torch_tensor=True,
                                                                        device=self.device, use_learnable_sigma=use_sigmas)
                if use_sigmas:
                    model_onset_kernel_preds = kernel_utils.generate_kernel_preds_sigma_gpu(preds_raw[0, :], sigmas_array=preds_raw[2, :],
                                                                                            device=self.device,
                                                                                            kernel_generating_function=kernel_utils.generate_kernel_preds_sigmas)
                    model_wakeup_kernel_preds = kernel_utils.generate_kernel_preds_sigma_gpu(preds_raw[1, :], sigmas_array=preds_raw[3, :],
                                                                                            device=self.device,
                                                                                            kernel_generating_function=kernel_utils.generate_kernel_preds_sigmas)
                else:
                    model_onset_kernel_preds = kernel_utils.generate_kernel_preds_gpu(preds_raw[0, :], device=self.device,
                                                                                      kernel_generating_function=kernel_utils.generate_kernel_preds,
                                                                                      kernel_radius=kernel_size, max_clip=2 * pred_width + 5 * kernel_size)
                    model_wakeup_kernel_preds = kernel_utils.generate_kernel_preds_gpu(preds_raw[1, :], device=self.device,
                                                                                      kernel_generating_function=kernel_utils.generate_kernel_preds,
                                                                                      kernel_radius=kernel_size, max_clip=2 * pred_width + 5 * kernel_size)

            if onset_kernel_values is None:
                onset_kernel_values = model_onset_kernel_preds
                wakeup_kernel_values = model_wakeup_kernel_preds
            else:
                onset_kernel_values += model_onset_kernel_preds
                wakeup_kernel_values += model_wakeup_kernel_preds

        # average
        onset_kernel_values /= num_regression_models
        wakeup_kernel_values /= num_regression_models
        kernel_values_computation_time = time.time() - ctime

        # get the locations
        ctime = time.time()
        onset_locs = (onset_kernel_values[1:-1] > onset_kernel_values[0:-2]) & (onset_kernel_values[1:-1] > onset_kernel_values[2:])
        onset_locs = np.argwhere(onset_locs).flatten() + 1
        wakeup_locs = (wakeup_kernel_values[1:-1] > wakeup_kernel_values[0:-2]) & (wakeup_kernel_values[1:-1] > wakeup_kernel_values[2:])
        wakeup_locs = np.argwhere(wakeup_locs).flatten() + 1

        onset_values = onset_kernel_values[onset_locs]
        wakeup_values = wakeup_kernel_values[wakeup_locs]

        # prune and align
        onset_locs = onset_locs[onset_values > regression_cutoff] # prune low (kernel) confidence values, and nearby values
        wakeup_locs = wakeup_locs[wakeup_values > regression_cutoff]
        if len(onset_locs) > 0:
            onset_locs = postprocessing.prune(onset_locs, onset_values[onset_values > regression_cutoff], regression_pruning)
        if len(wakeup_locs) > 0:
            wakeup_locs = postprocessing.prune(wakeup_locs, wakeup_values[wakeup_values > regression_cutoff], regression_pruning)

        first_zero = postprocessing.compute_first_zero(secs_corr) # align to 15s and 45s
        if len(onset_locs) > 0:
            onset_locs = postprocessing.align_predictions(onset_locs, onset_kernel_values, first_zero=first_zero)
        if len(wakeup_locs) > 0:
            wakeup_locs = postprocessing.align_predictions(wakeup_locs, wakeup_kernel_values, first_zero=first_zero)
        first_postprocessing_time = time.time() - ctime

        ## now compute confidence
        ctime = time.time()
        onset_confidence_probas, wakeup_confidence_probas = None, None
        num_confidence_models = 0
        for confidence_model_pkg in self.confidence_models:
            model = confidence_model_pkg["model"]
            model_name = confidence_model_pkg["model_name"]
            prediction_length = confidence_model_pkg["prediction_length"]
            expand = confidence_model_pkg["expand"]
            use_time_information = confidence_model_pkg["use_time_information"]
            stride_count = confidence_model_pkg["stride_count"]
            input_data = confidence_model_pkg["input_data"]

            if models_subset is not None and model_name not in models_subset:
                continue
            num_confidence_models += 1

            with torch.no_grad():
                if use_time_information:
                    times = {"hours": hours_corr, "mins": mins_corr, "secs": secs_corr}
                else:
                    times = None

                if input_data == "anglez":
                    accel_data = accel_data_anglez
                elif input_data == "enmo":
                    accel_data = accel_data_enmo
                else:
                    accel_data = np.concatenate(
                        [accel_data_anglez, accel_data_enmo], axis=1
                    )

                _, onset_locs_probas, wakeup_locs_probas = model_event_density_unet.event_density_inference(model=model,
                                                                 time_series=accel_data,
                                                                 predicted_locations=[{
                                                                     "onset": onset_locs,
                                                                     "wakeup": wakeup_locs
                                                                 }],
                                                                 batch_size=batch_size,
                                                                 prediction_length=prediction_length,
                                                                 expand=expand,
                                                                 times=times,
                                                                 stride_count=stride_count,
                                                                 flip_augmentation=False,
                                                                 use_time_input=use_time_information,
                                                                 device=self.device)

            if onset_confidence_probas is None:
                onset_confidence_probas = onset_locs_probas[0]
                wakeup_confidence_probas = wakeup_locs_probas[0]
            else:
                onset_confidence_probas += onset_locs_probas[0]
                wakeup_confidence_probas += wakeup_locs_probas[0]
        onset_confidence_probas /= num_confidence_models
        wakeup_confidence_probas /= num_confidence_models

        confidence_computation_time = time.time() - ctime

        ## prune using cutoff
        ctime = time.time()
        if confidence_cutoff > 0:
            onset_locs = onset_locs[onset_confidence_probas > confidence_cutoff]
            onset_confidence_probas = onset_confidence_probas[onset_confidence_probas > confidence_cutoff]
            wakeup_locs = wakeup_locs[wakeup_confidence_probas > confidence_cutoff]
            wakeup_confidence_probas = wakeup_confidence_probas[wakeup_confidence_probas > confidence_cutoff]

        ## do augmentation now
        onset_locs, onset_confidence_probas = postprocessing.get_augmented_predictions_density(onset_locs, onset_kernel_values,
                                                                                onset_confidence_probas,
                                                                                cutoff_thresh=confidence_aug_cutoff)
        wakeup_locs, wakeup_confidence_probas = postprocessing.get_augmented_predictions_density(wakeup_locs, wakeup_kernel_values,
                                                                                  wakeup_confidence_probas,
                                                                                  cutoff_thresh=confidence_aug_cutoff)
        second_postprocessing_time = time.time() - ctime

        ## matrix values prune if necessary
        if use_matrix_profile_pruning:
            ctime = time.time()
            if len(onset_locs) > 0 or len(wakeup_locs) > 0:
                # compute matrix values
                anglez = accel_data_anglez[0, :].astype(np.float64)

                # compute left pad and downsample by pooling
                original_length = len(anglez)
                left_pad = original_length % downsampling_rate
                if left_pad != 0:
                    left_pad = downsampling_rate - left_pad
                if left_pad > 0:
                    anglez = np.pad(anglez, (left_pad, 0))
                anglez = anglez.reshape(-1, downsampling_rate).mean(axis=1)

                # compute matrix profile
                if self.stumpy_device is None:
                    self.stumpy_device = [device.id for device in numba.cuda.list_devices()]
                matrix_profile = stumpy.gpu_stump(anglez.astype(np.float64), m=4320 // downsampling_rate,
                                                  device_id=self.stumpy_device[0])[:, 0].astype(np.float32)
                matrix_profile = np.pad(matrix_profile, (0, 4320 // downsampling_rate - 1), mode="constant",
                                        constant_values=matrix_profile[-1])  # subsequence stride right

                # upsample and revert
                matrix_profile = np.repeat(matrix_profile, downsampling_rate)
                if left_pad > 0:
                    matrix_profile = matrix_profile[left_pad:]
                assert len(matrix_profile) == original_length
                assert matrix_profile.dtype == np.float32

                if len(onset_locs) > 0:
                    onset_locs_restrict = postprocessing.prune_matrix_profile(onset_locs, matrix_profile,
                                                                              return_idx=True)
                    onset_locs = onset_locs[onset_locs_restrict]
                    onset_confidence_probas = onset_confidence_probas[onset_locs_restrict]
                if len(wakeup_locs) > 0:
                    wakeup_locs_restrict = postprocessing.prune_matrix_profile(wakeup_locs, matrix_profile,
                                                                               return_idx=True)
                    wakeup_locs = wakeup_locs[wakeup_locs_restrict]
                    wakeup_confidence_probas = wakeup_confidence_probas[wakeup_locs_restrict]
            matrix_profile_pruning_time = time.time() - ctime
        else:
            matrix_profile_pruning_time = 0.0

        avg_kernel_values_time = kernel_values_computation_time / num_regression_models
        avg_confidence_time = confidence_computation_time / num_confidence_models

        time_elapsed_performance_metrics = {
            "kernel_values_computation_time": kernel_values_computation_time,
            "first_postprocessing_time": first_postprocessing_time,
            "matrix_profile_pruning_time": matrix_profile_pruning_time,
            "confidence_computation_time": confidence_computation_time,
            "second_postprocessing_time": second_postprocessing_time,
            "avg_kernel_values_time": avg_kernel_values_time,
            "avg_confidence_time": avg_confidence_time,
        }

        ## return
        return onset_locs, onset_confidence_probas, wakeup_locs, wakeup_confidence_probas, time_elapsed_performance_metrics
