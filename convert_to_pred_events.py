import os
import numpy as np

def load_all_pred_events_into_dict(folder_name="regression_preds"):
    FOLDER = "inference_regression_statistics/{}".format(folder_name)

    all_data = {}
    all_series_ids = [x.split(".")[0] for x in os.listdir("individual_train_series")]
    for series_id in all_series_ids:
        onset_labels = os.path.join(FOLDER, "{}_onset_locs.npy".format(series_id))
        wakeup_labels = os.path.join(FOLDER, "{}_wakeup_locs.npy".format(series_id))
        all_data[series_id] = {
            "onset": np.load(onset_labels),
            "wakeup": np.load(wakeup_labels)
        }
    return all_data

if __name__ == "__main__":
    load_all_pred_events_into_dict()