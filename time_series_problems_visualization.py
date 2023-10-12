import sys
import os
import pandas as pd
import numpy as np
from PySide2.QtWidgets import QApplication, QVBoxLayout, QFileDialog, QWidget, QHBoxLayout, QPushButton, QListWidget, QTabWidget, QListWidgetItem, QSplitter, QSlider, QLabel, QCheckBox, QComboBox
from PySide2.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import tqdm

import convert_to_good_events
import cleaned_combine_pseudo_labels

class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axis = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)


    def plot_data(self, title, anglez, enmo, timestamp, probas, events):
        self.axis.clear()
        x = pd.to_datetime(timestamp)  # Automatically parses the timestamp
        y1 = anglez / 35.52 # std computed by check_series_properties.py
        y2 = enmo / 0.1018 # std computed by check_series_properties.py
        self.axis.plot(x, y1, label="anglez")
        self.axis.plot(x, y2, label="enmo")
        self.axis.plot(x, probas, label="probas")

        for event_time, event_type in events:
            color = "blue" if event_type == 1 else "red"
            self.axis.axvline(pd.to_datetime(event_time), color=color, alpha=0.5, linestyle="--")

        self.axis.set_title(title)
        self.axis.legend()
        self.canvas.draw()

class MainWidget(QWidget):
    def __init__(self, problem_intervals):
        super(MainWidget, self).__init__(None)
        self.problem_intervals = problem_intervals
        self.preloaded_problem_intervals = []
        self.preload_data()

        self.setWindowTitle("Visualization of errors")
        self.resize(1280, 720)

        self.layout = QVBoxLayout(self)

        self.display_widget = MatplotlibWidget()

        self.labels_widget = QWidget()
        self.labels_layout = QHBoxLayout(self.labels_widget)
        self.left_button = QPushButton("<")
        self.right_button = QPushButton(">")
        self.series_label = QLabel("0")
        self.labels_layout.addWidget(self.left_button)
        self.labels_layout.addWidget(self.series_label)
        self.labels_layout.addWidget(self.right_button)

        self.selection_slider = QSlider(Qt.Horizontal)
        self.selection_slider.setMinimum(0)
        self.selection_slider.setMaximum(len(problem_intervals) - 1)

        self.layout.addWidget(self.display_widget)
        self.layout.addWidget(self.labels_widget)
        self.layout.addWidget(self.selection_slider)

        self.selection_slider.valueChanged.connect(self.update_display)
        self.left_button.clicked.connect(self.left_button_clicked)
        self.right_button.clicked.connect(self.right_button_clicked)

    def preload_data(self):
        for series_id, start, end, labels in tqdm.tqdm(self.problem_intervals):
            anglez, enmo, timestamp, events = load_file(series_id, start, end)
            self.preloaded_problem_intervals.append((series_id, start, end, anglez, enmo, timestamp, labels, events))

    def left_button_clicked(self):
        self.selection_slider.setValue(max(self.selection_slider.value() - 1, 0))
        self.update_display()

    def right_button_clicked(self):
        self.selection_slider.setValue(min(self.selection_slider.value() + 1, len(self.problem_intervals) - 1))
        self.update_display()

    def update_display(self):
        selected_index = self.selection_slider.value()
        self.series_label.setText(str(selected_index))
        series_id, start, end, anglez, enmo, timestamp, labels, events = self.preloaded_problem_intervals[selected_index]
        self.display_widget.plot_data(series_id, anglez, enmo, timestamp, labels, events)

def load_file(item, start, end):
    filename = "./individual_train_series/" + item + ".parquet"
    df = pd.read_parquet(filename).iloc[start:end]
    events = load_extra_events(item, start, end)

    return df["anglez"], df["enmo"], df["timestamp"], events

def load_extra_events(series_id: str, start, end):
    events = pd.read_csv("data/train_events.csv")
    assert set(list(events["event"].unique())) == {"onset", "wakeup"}
    events = events.loc[events["series_id"] == series_id].dropna()
    events = events.loc[(events["step"] >= start) & (events["step"] <= end)]
    if len(events) == 0:
        return []

    events_list = []
    for k in range(len(events)):
        event = events.iloc[k]
        type = 1 if (event["event"] == "onset") else 2
        events_list.append((event["timestamp"], type))
    return events_list

def get_gt_series_metrics():
    recalls = []
    precisions = []
    ious = []

    problem_intervals = [] # tuple (series_id, start, end, predictions)
    all_series_ids = [series_id for series_id in os.listdir(convert_to_good_events.FOLDER) if series_id != "summary.txt"]

    for series_id in tqdm.tqdm(all_series_ids):
        intervals = []

        # load the gt intervals
        good_events_file = os.path.join(convert_to_good_events.FOLDER, series_id, "event.csv")
        try:
            intervals_dat = pd.read_csv(good_events_file, header=None).to_numpy(dtype="object")
            for k in range(intervals_dat.shape[0]):
                start, end, position = intervals_dat[k, :]
                intervals.append((start, end))
        except pd.errors.EmptyDataError:
            pass

        # load the pseudo labels
        pseudo_labels = np.load(os.path.join(cleaned_combine_pseudo_labels.LABELS_FOLDER, series_id + ".npy"))

        # compare with pseudo labels
        for start, end in intervals:
            start = int(start)
            end = int(end) + 1
            intersection = np.sum(pseudo_labels[start:end])
            recall = intersection / (end - start)

            positions = np.argwhere(pseudo_labels[start:end] == 1).flatten()
            pseudo_min = np.min(positions) + start
            pseudo_max = np.max(positions) + start

            while pseudo_min > 0 and pseudo_labels[pseudo_min] == 1:
                pseudo_min -= 1
            while pseudo_max < len(pseudo_labels) and pseudo_labels[pseudo_max] == 1:
                pseudo_max += 1
            pseudo_min += 1

            precision = intersection / (pseudo_max - pseudo_min)
            iou = intersection / (max(end, pseudo_max) - min(start, pseudo_min))

            # save metrics
            recalls.append(recall)
            precisions.append(precision)
            ious.append(iou)

            # save problem intervals
            if (recall < 0.95) or (precision < 0.95) or (iou < 0.95):
                length = end - start
                sstart = max(0, start - int(length * 1.1))
                send = min(len(pseudo_labels), end + int(length * 1.1))
                problem_intervals.append((series_id, sstart, send, pseudo_labels[sstart:send]))

    return recalls, precisions, ious, problem_intervals


if __name__ == "__main__":
    recalls, precisions, ious, problem_intervals = get_gt_series_metrics()

    print("Number of problem intervals: {} / {}".format(len(problem_intervals), len(recalls)))

    app = QApplication(sys.argv)

    # plot boxplot of recalls, precisions and ious
    fig, ax = plt.subplots()
    ax.boxplot([recalls, precisions, ious])
    ax.set_xticklabels(["Recall", "Precision", "IoU"])
    plt.show()

    main_widget = MainWidget(problem_intervals)
    main_widget.show()
    sys.exit(app.exec_())
