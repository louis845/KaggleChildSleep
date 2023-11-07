import json
import time
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide2.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QCheckBox, QSlider, QLabel, QSplitter, QHBoxLayout
from PySide2.QtCore import Qt

import convert_to_npy_naive
import metrics_iou

class MainWindow(QMainWindow):
    def __init__(self, data):
        super(MainWindow, self).__init__()
        self.prev_time = time.time()
        self.data = data

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.checkboxes = [QCheckBox(d["name"]) for d in sorted(self.data, key=lambda x: x["name"])]
        for checkbox in self.checkboxes:
            checkbox.stateChanged.connect(self.plot_data)

        self.prob_slider = QSlider(Qt.Horizontal)
        self.prob_slider.setRange(0, 100)
        self.prob_slider.valueChanged.connect(self.plot_data)

        self.tol_slider = QSlider(Qt.Horizontal)
        self.tol_slider.setRange(20, 60)
        self.tol_slider.valueChanged.connect(self.plot_data)

        self.prob_label = QLabel()
        self.tol_label = QLabel()

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        main_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(main_splitter)

        main_splitter.addWidget(self.canvas)

        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout()
        bottom_widget.setLayout(bottom_layout)

        checkbox_widget = QWidget()
        checkbox_layout = QVBoxLayout()
        checkbox_widget.setLayout(checkbox_layout)
        for checkbox in self.checkboxes:
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

        main_splitter.addWidget(bottom_widget)

    def plot_data(self):
        selected_list = [d["folder_name"] for d in self.data if self.checkboxes[self.data.index(d)].isChecked()]
        probability_threshold = self.prob_slider.value() / 100.0
        tol_radius = self.tol_slider.value()

        self.prob_label.setText(f'Probability threshold: {probability_threshold}')
        self.tol_label.setText(f'Tolerance radius: {tol_radius} (mins)')

        if len(selected_list) == 0:
            return
        if time.time() - self.prev_time > 0.5:
            self.prev_time = time.time()

            data_folder = os.path.join("./inference_confidence_statistics/confidence_labels",
                                       metric_data["folder_name"])
            for series_id in all_series_ids:
                onset_preds = np.load(os.path.join(data_folder, "{}_onset.npy".format(series_id)))
                wakeup_preds = np.load(os.path.join(data_folder, "{}_wakeup.npy".format(series_id)))

            self.figure.clear()
            ax = self.figure.add_subplot(111)

            self.canvas.draw()


if __name__ == '__main__':
    all_series_ids = [x.split(".")[0] for x in os.listdir("./individual_train_series")]

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
            per_seriesid_events["series_id"]["onset"].extend(onsets.values)
        if len(wakeups) > 0:
            per_seriesid_events["series_id"]["wakeup"].extend(wakeups.values)

    with open("./inference_confidence_statistics/inference_confidence_preds_options.json") as f:
        metrics_data = json.load(f)


    app = QApplication([])
    win = MainWindow(data=metrics_data)
    win.show()
    app.exec_()
