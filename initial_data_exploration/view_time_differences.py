import os

import tqdm
import pyarrow.parquet as pq
import pandas as pd

def analyze_parquet_file(parquet_path):
    # Read Parquet file into a pandas DataFrame
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    diffs = df["timestamp"].diff()[1:]

    return diffs

def analyze_parquet_file_localized(parquet_path):
    # Read Parquet file into a pandas DataFrame
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).apply(lambda dt: dt.tz_localize(None))
    diffs = df["timestamp"].diff()[1:]

    return diffs

all_time_diffs = set()
all_time_diffs_localized = set()
for file in tqdm.tqdm(os.listdir("../individual_train_series")):
    df = pq.read_table(os.path.join("../individual_train_series", file)).to_pandas()
    print(df["timestamp"].iloc[0])
    """time_diffs = analyze_parquet_file(os.path.join("../individual_train_series", file))
    time_diffs_localized = analyze_parquet_file_localized(os.path.join("../individual_train_series", file))
    all_time_diffs.update(set(time_diffs.unique()))
    all_time_diffs_localized.update(set(time_diffs_localized.unique()))"""

# Output: {Timedelta('0 days 00:00:05')}
print(all_time_diffs)
# Output: {Timedelta('0 days 01:00:05'), Timedelta('-1 days +23:00:05'), Timedelta('0 days 00:00:05')}
print(all_time_diffs_localized)