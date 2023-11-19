import os
import shutil

import tqdm
import pandas as pd

regression_models_5cv = ["regress_standard5cv_model_fold{}", "regress_standard5cv_mid_model_fold{}", "regress_standard5cv_wide_model_fold{}"]
regression_models = ["regress_combined_model", "regress_combined_mid_model", "regress_combined_wide_model"]

confidence_models_5cv = ["event5fold_2elastic_length3_drop_model_fold{}", "event5fold_2elastic_length2_time_drop_model_fold{}"]

out_folder = "final_models"

if __name__ == "__main__":
    # copy 5cv regression models
    for model in tqdm.tqdm(regression_models_5cv):
        for k in range(1, 6):
            model_name = model.format(k)
            model_dir = os.path.join("models", model.format(k))

            shutil.copy(os.path.join(model_dir, "model.pt"), os.path.join(out_folder, model_name + ".pt"))

    # copy regression models (trained on all data)
    for model in tqdm.tqdm(regression_models):
        model_dir = os.path.join("models", model)

        shutil.copy(os.path.join(model_dir, "model.pt"), os.path.join(out_folder, model + ".pt"))

    # copy 5cv confidence models
    for model in tqdm.tqdm(confidence_models_5cv):
        for k in range(1, 6):
            # usual model
            model_name = model.format(k)
            model_dir = os.path.join("models", model.format(k))

            shutil.copy(os.path.join(model_dir, "model.pt"), os.path.join(out_folder, model_name + ".pt"))

            # best model (pick early stopped best epoch)
            val_metrics = pd.read_csv(os.path.join(model_dir, "val_metrics.csv"), index_col=0)
            val_mAP = val_metrics["val_onset_mAP"] + val_metrics["val_wakeup_mAP"]
            best_model_idx = int(val_mAP.iloc[:30].idxmax())
            shutil.copy(os.path.join(model_dir, "model_{}.pt".format(best_model_idx)), os.path.join(out_folder, model_name + "_best.pt"))