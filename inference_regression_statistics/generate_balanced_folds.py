import os
import json

with open("generated_statistics/inference_regression_statistics_per_series_id.json", "r") as f:
    stats_per_seriesid = json.load(f)

onset_aes = stats_per_seriesid["Standard 5CV"]["onset_small_aes"]
wakeup_aes = stats_per_seriesid["Standard 5CV"]["wakeup_small_aes"]

all_series_ids = list(onset_aes.keys())

mean_aes = {}
for series_id in all_series_ids:
    mean_aes[series_id] = (onset_aes[series_id] + wakeup_aes[series_id]) / 2

mean_aes = {k: v for k, v in sorted(mean_aes.items(), key=lambda item: item[1])}

# Generate 5 folds
folds = [[], [], [], [], []]
for i, series_id in enumerate(mean_aes.keys()):
    folds[i % 5].append(series_id)

# Generate train/test splits
for i in range(5):
    train_series_ids = []
    for j in range(5):
        if j != i:
            train_series_ids = folds[j] + train_series_ids

    test_series_ids = folds[i]

    with open("../folds/balanced5cv_fold_1_train.json", "w") as f:
        json.dump({"dataset": train_series_ids}, f, indent=4)

    with open("../folds/balanced5cv_fold_1_test.json", "w") as f:
        json.dump({"dataset": test_series_ids}, f, indent=4)

    print("Fold {} train size: {}".format(i, len(train_series_ids)))
    print("Fold {} test size: {}".format(i, len(test_series_ids)))

print("Done.")
