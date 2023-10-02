import argparse
import json
import os

import numpy as np
import pandas as pd

folds_dataset_folder = "folds"
if not os.path.exists(folds_dataset_folder):
    os.makedirs(folds_dataset_folder)

def dataset_exists(dataset_name: str):
    return os.path.isfile(os.path.join(folds_dataset_folder, dataset_name + ".json"))

def load_dataset(dataset_name: str) -> list[str]:
    with open(os.path.join(folds_dataset_folder, dataset_name + ".json"), "r") as f:
        cfg = json.load(f)

    return cfg["dataset"]

def save_dataset(dataset_name: str, dataset: list[str]):
    with open(os.path.join(folds_dataset_folder, dataset_name + ".json"), "w") as f:
        json.dump({"dataset": dataset}, f, indent=4)

def add_argparse_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)

def parse_args(args: argparse.Namespace) -> tuple[list[str], list[str], str, str]:
    train_data_name = args.train_data
    val_data_name = args.val_data
    assert dataset_exists(train_data_name)
    assert dataset_exists(val_data_name)

    train_data = load_dataset(train_data_name)
    val_data = load_dataset(val_data_name)

    # assert that they are disjoint
    assert len(set(train_data).intersection(set(val_data))) == 0

    return train_data, val_data, train_data_name, val_data_name

if __name__ == "__main__":
    from sklearn.model_selection import KFold

    # you should run extract_individual_series.py before this
    assert os.path.isdir("individual_train_series"), "You should run extract_individual_series.py before this"

    all_series = [file.split(".")[0] for file in os.listdir("individual_train_series")]

    # 3 folds
    kf = KFold(n_splits=3, shuffle=True)
    for fold, (train_index, val_index) in enumerate(kf.split(all_series)):
        train_data = [all_series[index] for index in train_index]
        val_data = [all_series[index] for index in val_index]

        save_dataset(f"fold_{(fold + 1)}_train", train_data)
        save_dataset(f"fold_{(fold + 1)}_val", val_data)
