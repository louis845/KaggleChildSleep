import os
import json

import numpy as np
import matplotlib.pyplot as plt

def load_series_ids(fold_file):
    with open(fold_file, "r") as f:
        series_ids = json.load(f)["dataset"]

    return series_ids

folds_files = ["../folds/fold_{}_val_5cv.json".format(i) for i in range(1, 6)]
folds_series_ids = [load_series_ids(fold_file) for fold_file in folds_files]

preds_folder = "../inference_combined_statistics/combined_predictions/{}/width50".format("event5fold_time_2length_best")
regression_preds_folder = "../inference_regression_statistics/regression_preds"

threshold = 0.2

differences_per_fold = {}
restricted_differences_per_fold = {}
counts_per_fold = {}
restricted_counts_per_fold = {}

for fold in range(1, 6):
    fold_series_ids = folds_series_ids[fold - 1]
    differences = []
    counts = 0
    for series_id in fold_series_ids:
        pred_file = os.path.join(preds_folder, "{}_wakeup.npy".format(series_id))
        preds = np.load(pred_file)

        preds_locs_file = os.path.join(regression_preds_folder, "{}_wakeup_locs.npy".format(series_id))
        preds_locs = np.load(preds_locs_file)

        idxs_above_thresh = preds[preds_locs] >= threshold
        preds_locs = preds_locs[idxs_above_thresh]
        counts += len(preds_locs)

        if len(preds_locs) > 1:
            differences.extend(np.diff(preds_locs))

    differences = np.array(differences)
    restricted_differences = differences[differences < 8640]

    restricted_differences_per_fold["Fold " + str(fold)] = restricted_differences
    differences_per_fold["Fold " + str(fold)] = differences
    counts_per_fold[fold] = counts
    restricted_counts_per_fold[fold] = len(restricted_differences)

# plot head value per fold
fig, ax = plt.subplots()
ax.boxplot(differences_per_fold.values())
ax.set_xticklabels(["Fold {}\n(Count: {})".format(i, counts_per_fold[i]) for i in range(1, 6)])
ax.set_ylabel("Differences")
ax.set_xlabel("Fold")
ax.set_title("Differences per fold")
plt.show()

fig, ax = plt.subplots()
ax.boxplot(restricted_differences_per_fold.values())
ax.set_xticklabels(["Fold {}\n(Count: {})".format(i, restricted_counts_per_fold[i]) for i in range(1, 6)])
ax.set_ylabel("Restricted Differences")
ax.set_xlabel("Fold")
ax.set_title("Restricted Differences per fold")
plt.show()

# plot cumulative distribution of head value per fold. there should be 5 plots, one for each fold
# the x-axis should be the percentile (0-100) and the y-axis should be the head value

"""fig, ax = plt.subplots()
for fold in range(1, 6):
    fold_head_values = head_value_per_fold["Fold " + str(fold)]
    fold_head_values = np.sort(fold_head_values)

    ax.plot(np.linspace(0, 100, len(fold_head_values)), fold_head_values, label="Fold " + str(fold))

ax.set_xlabel("Percentile")
ax.set_ylabel("Head value")
ax.set_title("Cumulative distribution of head value per fold")
ax.legend()
plt.show()"""

print(sum(counts_per_fold.values()))
import pandas as pd
x = pd.read_csv("../data/train_events.csv").dropna()
x = x.loc[x["event"] == "wakeup"]
print(len(x))