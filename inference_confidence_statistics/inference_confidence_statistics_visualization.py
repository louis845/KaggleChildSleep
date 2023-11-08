import json
import time
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.widgets import RectangleSelector
from PySide2.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QCheckBox, QSlider, QLabel, QSplitter, QPushButton
from PySide2.QtCore import Qt

import convert_to_npy_naive
import metrics_iou

class MainWindow(QMainWindow):
    def __init__(self, data):
        super(MainWindow, self).__init__()
        self.prev_time = time.time()
        self.data = sorted(data, key=lambda x: x["name"])

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.checkboxes = {d["name"]: QCheckBox(d["name"]) for d in self.data}

        self.prob_slider = QSlider(Qt.Horizontal)
        self.prob_slider.setRange(0, 100)
        self.prob_slider.valueChanged.connect(self.update_sliders_text)

        self.tol_slider = QSlider(Qt.Horizontal)
        self.tol_slider.setRange(20, 60)
        self.tol_slider.valueChanged.connect(self.update_sliders_text)

        self.prob_label = QLabel()
        self.tol_label = QLabel()

        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.plot_data)

        self.init_ui()

        self.rs = None

        self.onsets_tpr, self.onsets_fpr = {}, {}
        self.wakeups_tpr, self.wakeups_fpr = {}, {}

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        main_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(main_splitter)

        plot_widget = QWidget()
        plot_layout = QVBoxLayout()
        plot_widget.setLayout(plot_layout)
        plot_layout.addWidget(self.canvas)
        toolbar = NavigationToolbar2QT(self.canvas, plot_widget)
        plot_layout.addWidget(toolbar)

        main_splitter.addWidget(plot_widget)

        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout()
        bottom_widget.setLayout(bottom_layout)

        checkbox_widget = QWidget()
        checkbox_layout = QVBoxLayout()
        checkbox_widget.setLayout(checkbox_layout)
        for checkbox in self.checkboxes.values():
            checkbox_layout.addWidget(checkbox)

        slider_widget = QWidget()
        slider_layout = QVBoxLayout()
        slider_widget.setLayout(slider_layout)

        # Set initial text of the labels and align the text to be center
        self.prob_label.setText(f'Probability threshold: {self.prob_slider.value() / 100.0}')
        self.prob_label.setAlignment(Qt.AlignCenter)
        self.tol_label.setText(f'Tolerance radius: {self.tol_slider.value()} (mins)')
        self.tol_label.setAlignment(Qt.AlignCenter)

        slider_layout.addWidget(self.prob_label)
        slider_layout.addWidget(self.prob_slider)
        slider_layout.addWidget(self.tol_label)
        slider_layout.addWidget(self.tol_slider)

        bottom_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter.addWidget(checkbox_widget)
        bottom_splitter.addWidget(slider_widget)

        bottom_layout.addWidget(bottom_splitter)
        bottom_layout.addWidget(self.plot_button)

        main_splitter.addWidget(bottom_widget)

    def get_sliders_values(self):
        probability_threshold = self.prob_slider.value() / 100.0
        tol_radius = self.tol_slider.value()
        return probability_threshold, tol_radius

    def update_sliders_text(self):
        probability_threshold, tol_radius = self.get_sliders_values()
        self.prob_label.setText(f'Probability threshold: {probability_threshold}')
        self.tol_label.setText(f'Tolerance radius: {tol_radius} (mins)')

    def plot_data(self):
        selected_list = [data_entry for data_entry in self.data if self.checkboxes[data_entry["name"]].isChecked()]
        probability_threshold, tol_radius = self.get_sliders_values()

        if len(selected_list) == 0:
            return
        if time.time() - self.prev_time > 0.5:
            self.prev_time = time.time()

            self.figure.clear()
            ax = self.figure.add_subplot(111)

            self.onsets_tpr.clear()
            self.onsets_fpr.clear()
            self.wakeups_tpr.clear()
            self.wakeups_fpr.clear()
            for k in range(len(selected_list)):
                metric_data = selected_list[k]

                folder_name = metric_data["folder_name"]
                name = metric_data["name"]

                onsets_tpr, onsets_fpr = np.zeros(len(all_series_ids)), np.zeros(len(all_series_ids))
                wakeups_tpr, wakeups_fpr = np.zeros(len(all_series_ids)), np.zeros(len(all_series_ids))
                data_folder = os.path.join("./inference_confidence_statistics/confidence_labels", folder_name)
                for i, series_id in enumerate(all_series_ids):
                    onset_preds = np.load(os.path.join(data_folder, "{}_onset.npy".format(series_id)))
                    wakeup_preds = np.load(os.path.join(data_folder, "{}_wakeup.npy".format(series_id)))
                    tpr_onset, fpr_onset = metrics_iou.compute_tpr_fpr_metrics(onset_preds, per_seriesid_events[series_id]["onset"],
                                                        p_threshold=probability_threshold, tolerance_radius=tol_radius)
                    tpr_wakeup, fpr_wakeup = metrics_iou.compute_tpr_fpr_metrics(wakeup_preds, per_seriesid_events[series_id]["wakeup"],
                                                        p_threshold=probability_threshold, tolerance_radius=tol_radius)
                    onsets_tpr[i], onsets_fpr[i] = tpr_onset, fpr_onset
                    wakeups_tpr[i], wakeups_fpr[i] = tpr_wakeup, fpr_wakeup

                ax.scatter(onsets_tpr, onsets_fpr, label="Onset {}".format(name), s=10)
                ax.scatter(wakeups_tpr, wakeups_fpr, label="Wakeup {}".format(name), s=10)

                self.onsets_tpr[name] = onsets_tpr
                self.onsets_fpr[name] = onsets_fpr
                self.wakeups_tpr[name] = wakeups_tpr
                self.wakeups_fpr[name] = wakeups_fpr

            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xticks(np.arange(0.0, 1.1, 0.2))
            ax.set_yticks(np.arange(0.0, 1.1, 0.2))

            ax.set_xlabel("True Positive Rate")
            ax.set_ylabel("False Positive Rate")
            ax.set_title("Performance Scatter")
            ax.legend()

            self.rs = RectangleSelector(ax, self.onselect, use_data_coordinates=True)

            self.canvas.draw()

    def onselect(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        problem_series_set = []
        for name in self.onsets_tpr.keys():
            tpr_onset, fpr_onset = self.onsets_tpr[name], self.onsets_fpr[name]
            tpr_wakeup, fpr_wakeup = self.wakeups_tpr[name], self.wakeups_fpr[name]

            indices = np.where((tpr_onset >= x1) & (tpr_onset <= x2) & (fpr_onset >= y1) & (fpr_onset <= y2))[0]
            print("Onset {}:".format(name))
            print("Series ID(s): {}".format(all_series_ids[indices]))
            for k in indices:
                if all_series_ids[k] not in problem_series_set:
                    problem_series_set.append(all_series_ids[k])

            indices = np.where((tpr_wakeup >= x1) & (tpr_wakeup <= x2) & (fpr_wakeup >= y1) & (fpr_wakeup <= y2))[0]
            print("Wakeup {}:".format(name))
            print("Series ID(s): {}".format(all_series_ids[indices]))
            for k in indices:
                if all_series_ids[k] not in problem_series_set:
                    problem_series_set.append(all_series_ids[k])

            print()

        problem_series_set.sort()
        print("Problem series ID(s): {}".format(problem_series_set))
        print("---------------------------------------------------")
        print()
        print()



if __name__ == '__main__':
    all_series_ids = [x.split(".")[0] for x in os.listdir("./individual_train_series")]
    all_series_ids = np.array(all_series_ids, dtype="object")

    all_data = convert_to_npy_naive.load_all_data_into_dict()
    events = pd.read_csv("./data/train_events.csv")
    events = events.dropna()
    per_seriesid_events = {}
    for series_id in all_series_ids:
        per_seriesid_events[series_id] = {
            "onset": [], "wakeup": []
        }
        series_events = events.loc[events["series_id"] == series_id]
        onsets = series_events.loc[series_events["event"] == "onset"]["step"]
        wakeups = series_events.loc[series_events["event"] == "wakeup"]["step"]
        if len(onsets) > 0:
            per_seriesid_events[series_id]["onset"].extend(onsets.to_numpy(np.int32))
        if len(wakeups) > 0:
            per_seriesid_events[series_id]["wakeup"].extend(wakeups.to_numpy(np.int32))

    with open("./inference_confidence_statistics/inference_confidence_preds_options.json") as f:
        metrics_data = json.load(f)


    app = QApplication([])
    win = MainWindow(data=metrics_data)
    win.show()
    app.exec_()
