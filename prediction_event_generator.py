import torch
import numpy as np
import pandas as pd

def generate_event_predictions(pred_probas, tolerances: list[int]):
    if type(pred_probas) == torch.Tensor:
        pred_probas = pred_probas.cpu().numpy()
    tolerances = np.array(tolerances, dtype=np.int32)
    tolerances = np.sort(tolerances)[::-1]

    # pred_probas is a 2D array of shape (2, T)
    onset_probas = pred_probas[0, :]
    wakeup_probas = pred_probas[1, :]
    total_time = pred_probas.shape[1]

    # order the predictions by probability
    onset_order = np.argsort(onset_probas)
    wakeup_order = np.argsort(wakeup_probas)

    # get the top 1000 predictions
    pred_upper_bound = 1000
    onset_top = onset_order[-pred_upper_bound:]
    wakeup_top = wakeup_order[-pred_upper_bound:]

    # add iteratively
    onset_list = np.full(fill_value=-1, shape=(pred_upper_bound,), dtype=np.int32)
    wakeup_list = np.full(fill_value=-1, shape=(pred_upper_bound,), dtype=np.int32)
    onset_probas_list = np.full(fill_value=-1, shape=(pred_upper_bound,), dtype=np.float32)
    wakeup_probas_list = np.full(fill_value=-1, shape=(pred_upper_bound,), dtype=np.float32)
    num_onset_added = 0
    num_wakeup_added = 0

    for tolerance in tolerances:
        for j in range(pred_upper_bound - 1, -1, -1):
            onset_time = onset_top[j]
            wakeup_time = wakeup_top[j]

            if (num_onset_added == 0) or np.all(np.abs(onset_time - onset_list[:num_onset_added]) >= tolerance):
                onset_list[num_onset_added] = onset_time
                onset_probas_list[num_onset_added] = onset_probas[onset_time]
                num_onset_added += 1

            if (num_wakeup_added == 0) or np.all(np.abs(wakeup_time - wakeup_list[:num_wakeup_added]) >= tolerance):
                wakeup_list[num_wakeup_added] = wakeup_time
                wakeup_probas_list[num_wakeup_added] = wakeup_probas[wakeup_time]
                num_wakeup_added += 1

    # create events
    events = [] # tuple (time, proba, type)
    for i in range(num_onset_added):
        events.append((onset_list[i], onset_probas_list[i], "onset"))
    for i in range(num_wakeup_added):
        events.append((wakeup_list[i], wakeup_probas_list[i], "wakeup"))

    return events

class EventsRecorder:
    def __init__(self):
        self.recorded_events = {
            "series_id": [],
            "step": [],
            "event": [],
            "score": []
        }

    def record(self, series_id: str, pred_probas, tolerances: list[int]):
        events = generate_event_predictions(pred_probas, tolerances)
        for time, proba, event_type in events:
            self.recorded_events["series_id"].append(series_id)
            self.recorded_events["step"].append(time)
            self.recorded_events["event"].append(event_type)
            self.recorded_events["score"].append(proba)

    def clear_events(self):
        self.recorded_events["series_id"].clear()
        self.recorded_events["step"].clear()
        self.recorded_events["event"].clear()
        self.recorded_events["score"].clear()

    def convert_to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.recorded_events)
