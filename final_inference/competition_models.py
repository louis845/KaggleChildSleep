import json
import os
import time

import torch
import numpy as np

import postprocessing
import model_event_unet
import kernel_utils

class CompetitionModels:
    def __init__(self, model_config_file, models_root_dir: str, device: torch.device):
        with open(model_config_file, "r") as f:
            self.model_config = json.load(f)
        self.models_root_dir = models_root_dir
        self.device = device

        self.regression_models = []
        self.confidence_models = []
        self.iou_converter = None

    def load_regression_models(self):
        for cfg in self.model_config["regression_models"]:
            regression_cfg = {
                "hidden_blocks": [2, 2, 2, 2, 3],
                "hidden_channels": [4, 4, 8, 16, 32],
                "pred_width": 120,
                "kernel_size": 9,
                "use_sigmas": False
            }
            if "hidden_blocks" in cfg:
                regression_cfg["hidden_blocks"] = cfg["hidden_blocks"]
            if "hidden_channels" in cfg:
                regression_cfg["hidden_channels"] = cfg["hidden_channels"]
            if "pred_width" in cfg:
                regression_cfg["pred_width"] = cfg["pred_width"]
            if "use_sigmas" in cfg:
                regression_cfg["use_sigmas"] = cfg["use_sigmas"]
            regression_cfg["model_name"] = cfg["model_name"]

            blocks_length = len(regression_cfg["hidden_blocks"])
            target_multiple = 3 * (2 ** (blocks_length - 2))

            model = model_event_unet.EventRegressorUnet(use_learnable_sigma=regression_cfg["use_sigmas"],
                                                        blocks=regression_cfg["hidden_blocks"],
                                                        hidden_channels=regression_cfg["hidden_channels"])
            model.to(self.device)
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
                "expand": 8640
            }

            if "attention_blocks" in cfg:
                confidence_cfg["attention_blocks"] = cfg["attention_blocks"]
            if "attention_mode" in cfg:
                confidence_cfg["attention_mode"] = cfg["attention_mode"]
            if "stride_count" in cfg:
                confidence_cfg["stride_count"] = cfg["stride_count"]
            if "use_time_information" in cfg:
                confidence_cfg["use_time_information"] = cfg["use_time_information"]
            if "expand" in cfg:
                confidence_cfg["expand"] = cfg["expand"]
            confidence_cfg["model_name"] = cfg["model_name"]

            model = model_event_unet.EventConfidenceUnet(in_channels=1, # we use anglez only
                                                         attention_blocks=confidence_cfg["attention_blocks"],
                                                         attention_mode=confidence_cfg["attention_mode"],
                                                         use_time_input=confidence_cfg["use_time_information"],
                                                         expected_attn_input_length=17280 + 2 * confidence_cfg["expand"])
            model.to(self.device)
            model.load_state_dict(torch.load(os.path.join(self.models_root_dir,
                                                          "{}.pt".format(confidence_cfg["model_name"])
                                                          ), weights_only=True, map_location=self.device))
            model.eval()

            model_pkg = {
                "model": model,
                "model_name": confidence_cfg["model_name"],
                "prediction_length": 17280,
                "expand": confidence_cfg["expand"],
                "use_time_information": confidence_cfg["use_time_information"],
                "stride_count": confidence_cfg["stride_count"]
            }

            self.confidence_models.append(model_pkg)

    def initialize_IOU_converter(self):
        self.iou_converter = model_event_unet.ProbasIOUScoreConverter(intersection_width=30 * 12, union_width=50 * 12, device=self.device)

    def load_models(self):
        self.load_regression_models()
        self.load_confidence_models()
        self.initialize_IOU_converter()

    def run_inference(self, series_id: str, accel_data: np.ndarray,
                      secs_corr: np.ndarray, mins_corr: np.ndarray, hours_corr: np.ndarray,
                      models_subset: list = None):
        ## cfg values for regression
        cutoff = 4.5
        pruning = 60

        ## find regression locations first
        ctime = time.time()
        onset_kernel_values = None
        wakeup_kernel_values = None
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

            with torch.no_grad():
                preds_raw = model_event_unet.event_regression_inference(model, accel_data, target_multiple=target_multiple, return_torch_tensor=True)
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
        onset_kernel_values /= len(self.regression_models)
        wakeup_kernel_values /= len(self.regression_models)
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
        onset_locs = onset_locs[onset_values > cutoff] # prune low (kernel) confidence values, and nearby values
        wakeup_locs = wakeup_locs[wakeup_values > cutoff]
        if len(onset_locs) > 0:
            onset_locs = postprocessing.prune(onset_locs, onset_values[onset_values > cutoff], pruning)
        if len(wakeup_locs) > 0:
            wakeup_locs = postprocessing.prune(wakeup_locs, wakeup_values[wakeup_values > cutoff], pruning)

        first_zero = postprocessing.compute_first_zero(secs_corr) # align to 15s and 45s
        if len(onset_locs) > 0:
            onset_locs = postprocessing.align_predictions(onset_locs, onset_kernel_values, first_zero=first_zero)
        if len(wakeup_locs) > 0:
            wakeup_locs = postprocessing.align_predictions(wakeup_locs, wakeup_kernel_values, first_zero=first_zero)
        first_postprocessing_time = time.time() - ctime

        ## cfg values for confidence
        batch_size = 512
        iou_averaging = ("iou_averaging" in self.model_config) and self.model_config["iou_averaging"]

        ## now compute confidence
        if iou_averaging:
            ctime = time.time()
            onset_IOU_probas, wakeup_IOU_probas = None, None
            for confidence_model_pkg in self.confidence_models:
                model = confidence_model_pkg["model"]
                model_name = confidence_model_pkg["model_name"]
                prediction_length = confidence_model_pkg["prediction_length"]
                expand = confidence_model_pkg["expand"]
                use_time_information = confidence_model_pkg["use_time_information"]
                stride_count = confidence_model_pkg["stride_count"]

                if models_subset is not None and model_name not in models_subset:
                    continue

                with torch.no_grad():
                    if use_time_information:
                        times = {"hours": hours_corr, "mins": mins_corr, "secs": secs_corr}
                    else:
                        times = None

                    preds = model_event_unet.event_confidence_inference(model=model, time_series=accel_data,
                                                                        batch_size=batch_size,
                                                                        prediction_length=prediction_length,
                                                                        expand=expand, times=times,
                                                                        stride_count=stride_count,
                                                                        flip_augmentation=False)

                onset_confidence_probas = preds[0, :]
                wakeup_confidence_probas = preds[1, :]

                # convert to IOU score
                model_onset_IOU_probas = self.iou_converter.convert(onset_confidence_probas)
                model_wakeup_IOU_probas = self.iou_converter.convert(wakeup_confidence_probas)

                if onset_IOU_probas is None:
                    onset_IOU_probas = model_onset_IOU_probas
                    wakeup_IOU_probas = model_wakeup_IOU_probas
                else:
                    onset_IOU_probas += model_onset_IOU_probas
                    wakeup_IOU_probas += model_wakeup_IOU_probas

            onset_IOU_probas /= len(self.confidence_models)
            wakeup_IOU_probas /= len(self.confidence_models)

            confidence_computation_time = time.time() - ctime
        else:
            ctime = time.time()
            onset_confidence_probas, wakeup_confidence_probas = None, None
            for confidence_model_pkg in self.confidence_models:
                model = confidence_model_pkg["model"]
                prediction_length = confidence_model_pkg["prediction_length"]
                expand = confidence_model_pkg["expand"]
                use_time_information = confidence_model_pkg["use_time_information"]
                stride_count = confidence_model_pkg["stride_count"]

                with torch.no_grad():
                    if use_time_information:
                        times = {"hours": hours_corr, "mins": mins_corr, "secs": secs_corr}
                    else:
                        times = None

                    preds = model_event_unet.event_confidence_inference(model=model, time_series=accel_data,
                                                                        batch_size=batch_size,
                                                                        prediction_length=prediction_length,
                                                                        expand=expand, times=times,
                                                                        stride_count=stride_count,
                                                                        flip_augmentation=False)

                if onset_confidence_probas is None:
                    onset_confidence_probas = preds[0, :]
                    wakeup_confidence_probas = preds[1, :]
                else:
                    onset_confidence_probas += preds[0, :]
                    wakeup_confidence_probas += preds[1, :]
            onset_confidence_probas /= len(self.confidence_models)
            wakeup_confidence_probas /= len(self.confidence_models)

            # convert to IOU score
            onset_IOU_probas = self.iou_converter.convert(onset_confidence_probas)
            wakeup_IOU_probas = self.iou_converter.convert(wakeup_confidence_probas)
            confidence_computation_time = time.time() - ctime

        ## do augmentation now
        ctime = time.time()
        onset_locs, onset_IOU_probas = postprocessing.get_augmented_predictions(onset_locs, onset_kernel_values,
                                                                                onset_IOU_probas, cutoff_thresh=0.01)
        wakeup_locs, wakeup_IOU_probas = postprocessing.get_augmented_predictions(wakeup_locs, wakeup_kernel_values,
                                                                                wakeup_IOU_probas, cutoff_thresh=0.01)
        second_postprocessing_time = time.time() - ctime

        avg_kernel_values_time = kernel_values_computation_time / len(self.regression_models)
        avg_confidence_time = confidence_computation_time / len(self.confidence_models)

        time_elapsed_performance_metrics = {
            "kernel_values_computation_time": kernel_values_computation_time,
            "first_postprocessing_time": first_postprocessing_time,
            "confidence_computation_time": confidence_computation_time,
            "second_postprocessing_time": second_postprocessing_time,
            "avg_kernel_values_time": avg_kernel_values_time,
            "avg_confidence_time": avg_confidence_time,
        }

        ## return
        return onset_locs, onset_IOU_probas, wakeup_locs, wakeup_IOU_probas, time_elapsed_performance_metrics
