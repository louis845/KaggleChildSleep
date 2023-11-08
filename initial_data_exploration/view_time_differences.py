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
    time_diffs = analyze_parquet_file(os.path.join("../individual_train_series", file))
    time_diffs_localized = analyze_parquet_file_localized(os.path.join("../individual_train_series", file))
    all_time_diffs.update(set(time_diffs.unique()))
    all_time_diffs_localized.update(set(time_diffs_localized.unique()))

print(all_time_diffs)
print(all_time_diffs_localized)