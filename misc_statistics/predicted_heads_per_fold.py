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

head_value_per_fold = {}
for fold in range(1, 6):
    fold_series_ids = folds_series_ids[fold - 1]
    head_values = []
    for series_id in fold_series_ids:
        pred_file = os.path.join(preds_folder, "{}_wakeup.npy".format(series_id))
        preds = np.load(pred_file)

        diff = preds[1:] - preds[:-1]
        increasing = diff >= 0

        head_value = None
        if not increasing[0]:
            head_value = preds[0]
        else:
            increasing_edge = increasing[:-1] & (~increasing[1:])
            first_decrease = np.argwhere(increasing_edge).flatten()[0]

            head_value = preds[first_decrease + 1]

        head_value = np.max(preds[:2160])

        head_values.append(head_value)

    head_value_per_fold["Fold " + str(fold)] = np.array(head_values)

# plot head value per fold
fig, ax = plt.subplots()
ax.boxplot(head_value_per_fold.values())
ax.set_xticklabels(head_value_per_fold.keys())
ax.set_ylabel("Head value")
ax.set_xlabel("Fold")
ax.set_title("Head value per fold")
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