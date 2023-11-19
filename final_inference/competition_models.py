import json
import os

import torch
import numpy as np

import model_event_unet

class CompetitionModels:
    def __init__(self, model_config_file, models_root_dir: str, device: torch.device):
        with open(model_config_file, "r") as f:
            self.model_config = json.load(f)
        self.models_root_dir = models_root_dir
        self.device = device

        self.regression_models = []
        self.confidence_models = []

    def load_regression_models(self):
        for cfg in self.model_config["regression_models"]:
            regression_cfg = {
                "hidden_blocks": [2, 2, 2, 2, 2],
                "hidden_channels": [4, 4, 8, 16, 32],
                "pred_width": 120,
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

            model = model_event_unet.EventRegressorUnet(use_learnable_sigma=regression_cfg["use_sigmas"],
                                                        blocks=regression_cfg["hidden_blocks"],
                                                        hidden_channels=regression_cfg["hidden_channels"])
            model.to(self.device)
            model.load_state_dict(torch.load(os.path.join(self.models_root_dir,
                                                          "{}.pt".format(regression_cfg["model_name"])
                                                          ), weights_only=True))
            model.eval()

            model_pkg = {
                "model": model,
                "pred_width": regression_cfg["pred_width"],
                "use_sigmas": regression_cfg["use_sigmas"]
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
                                                          ), weights_only=True))
            model.eval()

            model_pkg = {
                "model": model,
                "prediction_length": 17280,
                "expand": confidence_cfg["expand"],
                "use_time_information": confidence_cfg["use_time_information"],
                "stride_count": confidence_cfg["stride_count"]
            }

            self.confidence_models.append(model_pkg)

    def load_models(self):
        self.load_regression_models()
        self.load_confidence_models()

    def run_inference(self, series_id: np.ndarray, accel_data: np.ndarray,
                      secs_corr: np.ndarray, mins_corr: np.ndarray, hours_corr: np.ndarray):
        pass