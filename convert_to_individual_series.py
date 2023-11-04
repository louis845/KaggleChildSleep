import pandas as pd
import pyarrow.parquet as pq
import os
import tqdm

def extract_subtables(input_file, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    parquet_file = pq.ParquetFile(input_file)

    unique_series_ids = set()

    # Iterate over row groups to find unique series_id values
    for i in range(parquet_file.num_row_groups):
        df = parquet_file.read_row_group(i).to_pandas()
        unique_series_ids.update(df["series_id"].unique())

    # For each unique series_id, read only the rows where series_id equals the current id, and write to a new parquet file
    for series_id in tqdm.tqdm(unique_series_ids):
        df = pd.read_parquet(input_file, filters=[("series_id", "==", series_id)])
        df.to_parquet(os.path.join(output_dir, f"{series_id}.parquet"))

# Usage
extract_subtables("data/train_series.parquet", "individual_train_series")
