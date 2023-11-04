import pandas as pd

if __name__ == "__main__":
    events = pd.read_csv("../data/train_events.csv")

    onset_wakeup_inconsistencies = []

    for series_id in events["series_id"].unique():
        series_events = events.loc[events["series_id"] == series_id]
        assert len(series_events) % 2 == 0, "Series {} has an odd number of events".format(series_id)

        has_prev_onset = False
        for k in range(len(series_events)):
            event = series_events.iloc[k]

            # check whether onset and wakeup are alternating
            if k % 2 == 0:
                assert event["event"] == "onset", "Violated alternating pattern in series_id {}".format(series_id)
            else:
                assert event["event"] == "wakeup", "Violated alternating pattern in series_id {}".format(series_id)

            # check whether each each onset is followed by a wakeup
            has_event = pd.isna(event["step"])
            if k % 2 == 0:
                has_prev_onset = has_event
            else:
                if has_prev_onset != has_event:
                    onset_wakeup_inconsistencies.append(series_id)

    print("Onset/wakeup inconsistencies: {}".format(onset_wakeup_inconsistencies))