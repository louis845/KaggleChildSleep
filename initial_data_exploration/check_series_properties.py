import os.path

import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    lengths = []

    for file in tqdm.tqdm(os.listdir("../individual_train_series")):
        series_id = file.split(".")[0]
        series_info = pd.read_parquet(os.path.join("../individual_train_series", file))
        lengths.append(len(series_info))

    print("Max length: {}".format(np.max(lengths)))
    print("Min length: {}".format(np.min(lengths)))
    print("Mean length: {}".format(np.mean(lengths)))

    all_series_info = pd.read_parquet(os.path.join("../data", "train_series.parquet"))
    # compute the mean and variance of the anglez and enmo
    anglez_mean = all_series_info["anglez"].mean()
    anglez_std = all_series_info["anglez"].std()
    enmo_mean = all_series_info["enmo"].mean()
    enmo_std = all_series_info["enmo"].std()

    print("Anglez mean: {}".format(anglez_mean))
    print("Anglez std: {}".format(anglez_std))
    print("Enmo mean: {}".format(enmo_mean))
    print("Enmo std: {}".format(enmo_std))

    # plot the distribution of lengths
    sns.distplot(lengths, kde=False)
    plt.show()