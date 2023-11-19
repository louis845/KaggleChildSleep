import os

import numpy as np
import pandas as pd

def get_events_per_seriesid(train_events_file="./data/train_events.csv",
                            series_ids_folder="./individual_train_series"):
    all_series_ids = [x.split(".")[0] for x in os.listdir(series_ids_folder)]
    all_series_ids = np.array(all_series_ids, dtype="object")

    events = pd.read_csv(train_events_file)
    events = events.dropna()
    per_seriesid_events = {}
    for series_id in all_series_ids:
        per_seriesid_events[series_id] = {
            "onset": [], "wakeup": []
        }
        series_events = events.loc[events["series_id"] == series_id]
        onsets = series_events.loc[series_events["event"] == "onset"]["step"]
        wakeups = series_events.loc[series_events["event"] == "wakeup"]["step"]
        if len(onsets) > 0:
            per_seriesid_events[series_id]["onset"].extend(onsets.to_numpy(np.int32))
        if len(wakeups) > 0:
            per_seriesid_events[series_id]["wakeup"].extend(wakeups.to_numpy(np.int32))

    return per_seriesid_events
