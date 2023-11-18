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
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--val_data", type=str)
    parser.add_argument("--train_all", action="store_true")

def parse_args(args: argparse.Namespace) -> tuple[list[str], list[str], str, str]:
    train_data_name = args.train_data
    val_data_name = args.val_data
    train_all = args.train_all
    if not train_all:
        assert train_data_name is not None and val_data_name is not None, "You should specify train_data and val_data if you are not training on all data"
        assert dataset_exists(train_data_name)
        assert dataset_exists(val_data_name)

        train_data = load_dataset(train_data_name)
        val_data = load_dataset(val_data_name)

        # assert that they are disjoint
        assert len(set(train_data).intersection(set(val_data))) == 0
    else:
        all_series = [file.split(".")[0] for file in os.listdir("individual_train_series")]
        train_data = all_series.copy()
        val_data = all_series
        train_data_name = "all"
        val_data_name = "all"

    return train_data, val_data, train_data_name, val_data_name

if __name__ == "__main__":
    from sklearn.model_selection import KFold

    # you should run convert_to_individual_series.py before this
    assert os.path.isdir("individual_train_series"), "You should run convert_to_individual_series.py before this"

    all_series = [file.split(".")[0] for file in os.listdir("individual_train_series")]

    # 3 folds
    if not os.path.isfile("folds/fold_1_train.json"):
        kf = KFold(n_splits=3, shuffle=True)
        for fold, (train_index, val_index) in enumerate(kf.split(all_series)):
            train_data = [all_series[index] for index in train_index]
            val_data = [all_series[index] for index in val_index]

            save_dataset(f"fold_{(fold + 1)}_train", train_data)
            save_dataset(f"fold_{(fold + 1)}_val", val_data)

    # 5 folds, indicated by prefix 5
    if not os.path.isfile("folds/fold_1_train_5cv.json"):
        kf = KFold(n_splits=5, shuffle=True)
        for fold, (train_index, val_index) in enumerate(kf.split(all_series)):
            train_data = [all_series[index] for index in train_index]
            val_data = [all_series[index] for index in val_index]

            save_dataset(f"fold_{(fold + 1)}_train_5cv", train_data)
            save_dataset(f"fold_{(fold + 1)}_val_5cv", val_data)


    # 5 folds, 10 cv
    if not os.path.isfile("folds/fold_1_train_10cv.json"):
        kf = KFold(n_splits=5, shuffle=True)
        splits = []
        for (train_index, val_index) in kf.split(all_series):
            splits.append([all_series[index] for index in val_index])

        train_test_splits = [
            ((0, 1), (2, 3, 4)),
            ((0, 2), (1, 3, 4)),
            ((0, 3), (1, 2, 4)),
            ((0, 4), (1, 2, 3)),
            ((1, 2), (0, 3, 4)),
            ((1, 3), (0, 2, 4)),
            ((1, 4), (0, 2, 3)),
            ((2, 3), (0, 1, 4)),
            ((2, 4), (0, 1, 3)),
            ((3, 4), (0, 1, 2)),
        ]

        for fold, (train_subsets, val_subsets) in enumerate(train_test_splits):
            train_data = [series for subset in train_subsets for series in splits[subset]]
            val_data = [series for subset in val_subsets for series in splits[subset]]

            save_dataset(f"fold_{(fold + 1)}_train_10cv", train_data)
            save_dataset(f"fold_{(fold + 1)}_val_10cv", val_data)
