import pandas as pd
import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    events_df = pd.read_csv("data/train_events.csv")
    events_df = events_df.dropna()

    # check time consistency
    """
    for series_id in tqdm.tqdm(events_df["series_id"].unique()):
        series_pq = pd.read_parquet("individual_train_series/{}.parquet".format(series_id))

        series_event_df = events_df.loc[events_df["series_id"] == series_id]
        for k in range(len(series_event_df)):
            event = series_event_df.iloc[k]
            timestamp = event["timestamp"]
            assert str(timestamp) == str(series_pq.iloc[int(event["step"])]["timestamp"])"""

    # convert timestamp to datetime
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"]).apply(lambda dt: dt.tz_localize(None))

    print(events_df["timestamp"][0])

    events_df["time_in_day"] = events_df["timestamp"].dt.hour + events_df["timestamp"].dt.minute / 60.0 + events_df["timestamp"].dt.second / 3600.0

    # get events
    events_onset = events_df.loc[events_df["event"] == "onset"]
    events_wakeup = events_df.loc[events_df["event"] == "wakeup"]

    # create matplotlib figure with 3 plots, one for each event type, and one for all events.
    # they should plot the time distribution of the events, in 1hr bins
    fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)
    fig.suptitle("Time distribution of events")

    # plot onset events
    axs[0].hist(events_onset["time_in_day"], bins=24)
    axs[0].set_title("Onset events")

    # plot wakeup events
    axs[1].hist(events_wakeup["time_in_day"], bins=24)
    axs[1].set_title("Wakeup events")

    # plot all events
    axs[2].hist(events_df["time_in_day"], bins=24)
    axs[2].set_title("All events")

    # plot all events, but with a log scale
    axs[3].hist(events_df["time_in_day"], bins=24)
    axs[3].set_title("All events (log scale)")

    # show the plot
    plt.show()

    print(events_wakeup["time_in_day"].max())
    print(events_onset.loc[events_onset["time_in_day"] <= 20.4432]["time_in_day"].max())
