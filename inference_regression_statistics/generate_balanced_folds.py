import os
import json

with open("generated_statistics/inference_regression_statistics_per_series_id.json", "r") as f:
    stats_per_seriesid = json.load(f)

onset_aes = stats_per_seriesid["Standard 5CV"]["onset_small_aes"]
wakeup_aes = stats_per_seriesid["Standard 5CV"]["wakeup_small_aes"]

all_series_ids = list(onset_aes.keys())
all_actual_series_ids = [x.split(".")[0] for x in os.listdir("../individual_train_series")]
series_ids_set_diff = list(set(all_actual_series_ids).difference(set(all_series_ids)))

mean_aes = {}
for series_id in all_series_ids:
    mean_aes[series_id] = (onset_aes[series_id] + wakeup_aes[series_id]) / 2

mean_aes = {k: v for k, v in sorted(mean_aes.items(), key=lambda item: item[1])}

# Generate 10 folds
folds = [[], [], [], [], [], [], [], [], [], []]
for i, series_id in enumerate(mean_aes.keys()):
    folds[i % 10].append(series_id)
for j in range(len(series_ids_set_diff)):
    folds[(i + j + 1) % 10].append(series_ids_set_diff[j])

# Generate train/test splits
for i in range(10):
    train_series_ids = []
    for j in range(10):
        if j != i:
            train_series_ids = folds[j] + train_series_ids

    test_series_ids = folds[i]

    with open("../folds/balanced5cv_fold_{}_train.json".format(i + 1), "w") as f:
        json.dump({"dataset": train_series_ids}, f, indent=4)

    with open("../folds/balanced5cv_fold_{}_val.json".format(i + 1), "w") as f:
        json.dump({"dataset": test_series_ids}, f, indent=4)

    print("Fold {} train size: {}".format(i, len(train_series_ids)))
    print("Fold {} test size: {}".format(i, len(test_series_ids)))

print("Done.")
