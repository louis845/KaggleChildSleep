import os

import pyarrow.parquet as pq
import pandas as pd

def analyze_parquet_file(parquet_path):
    # Read Parquet file into a pandas DataFrame
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    diffs = df["timestamp"].diff()[1:]

    return diffs

all_time_diffs = set()
for file in os.listdir("individual_train_series"):
    time_diffs = analyze_parquet_file(os.path.join("individual_train_series", file))
    all_time_diffs.update(set(time_diffs.unique()))

print(all_time_diffs)