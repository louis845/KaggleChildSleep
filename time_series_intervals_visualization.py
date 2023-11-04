import sys
import os
import pandas as pd
import numpy as np
from PySide2.QtWidgets import QApplication, QVBoxLayout, QFileDialog, QWidget, QHBoxLayout, QPushButton, QListWidget, QTabWidget, QListWidgetItem, QSplitter, QSlider, QLabel, QCheckBox, QComboBox
from PySide2.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import tqdm

import inference_regression_preds
import inference_confidence_preds
import kernel_utils

def generate_kernel_preds(preds_array: np.ndarray):
    kernel_preds_array = np.zeros_like(preds_array)
    for k in tqdm.tqdm(range(len(preds_array))):
        kernel_utils.add_kernel(kernel_preds_array, k - float(preds_array[k]))
    return kernel_preds_array


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

        self.max_y = 10.0
        self.min_y = -10.0


    def plot_data(self, title, anglez, enmo, extras, timestamp, events):
        self.axis.clear()
        x = pd.to_datetime(timestamp)  # Automatically parses the timestamp
        y1 = anglez / 35.52 # std computed by check_series_properties.py
        y2 = enmo / 0.1018 # std computed by check_series_properties.py
        self.axis.set_ylim([self.min_y, self.max_y])
        self.axis.plot(x, y1, label="anglez")
        self.axis.plot(x, y2, label="enmo")
        #self.axis.plot(x, extras["onset"] / 100.0, label="onset")
        #self.axis.plot(x, extras["wakeup"] / 100.0, label="wakeup")
        #self.axis.plot(x, extras["onset_kernel"], label="onset_kernel")
        #self.axis.plot(x, extras["wakeup_kernel"], label="wakeup_kernel")
        self.axis.plot(x, extras["onset_conf"] * 10.0, label="onset_conf") # easier viewing
        self.axis.plot(x, extras["wakeup_conf"] * 10.0, label="wakeup_conf")

        for event_time, event_type in events:
            color = "blue" if event_type in [1, 3, 4] else "red"
            linestyle = "--" if event_type in [1, 2] else ":"
            self.axis.axvline(pd.to_datetime(event_time), color=color, alpha=0.5, linestyle=linestyle)

        self.axis.set_title(title)
        self.axis.legend()
        self.canvas.draw()

class MainWidget(QWidget):

    preloaded_intervals: list

    def __init__(self):
        super(MainWidget, self).__init__(None)
        self.preloaded_intervals = None
        self.events = pd.read_csv("data/train_events.csv")
        self.events = self.events.dropna()

        self.setWindowTitle("Visualization of time series intervals and events")
        self.resize(1280, 720)

        # Create layout
        self.layout = QVBoxLayout(self)

        self.main_widget = QWidget()
        self.series_id_items = QListWidget()
        self.left_right_splitter = QSplitter(Qt.Horizontal)
        self.left_right_splitter.addWidget(self.series_id_items)
        self.left_right_splitter.addWidget(self.main_widget)
        self.left_right_splitter.setSizes([0.2 * self.width(), 0.8 * self.width()])
        self.layout.addWidget(self.left_right_splitter)

        self.main_layout = QVBoxLayout(self.main_widget)

        self.display_widget = MatplotlibWidget()

        self.labels_widget = QWidget()
        self.labels_layout = QHBoxLayout(self.labels_widget)
        self.left_button = QPushButton("<")
        self.right_button = QPushButton(">")
        self.series_label = QLabel("0")
        self.series_label.setAlignment(Qt.AlignCenter)
        self.labels_layout.addWidget(self.left_button)
        self.labels_layout.addWidget(self.series_label)
        self.labels_layout.addWidget(self.right_button)
        self.labels_widget.setFixedHeight(40)

        self.selection_slider = QSlider(Qt.Horizontal)
        self.selection_slider.setMinimum(0)
        self.selection_slider.setMaximum(10)

        self.main_layout.addWidget(self.display_widget)
        self.main_layout.addWidget(self.labels_widget)
        self.main_layout.addWidget(self.selection_slider)

        # Add series ids
        series_ids = [x[:-8] for x in os.listdir("individual_train_series")]
        series_ids.sort()
        for series_id in series_ids:
            self.series_id_items.addItem(series_id)

        self.series_id_items.itemDoubleClicked.connect(self.preload_data)
        self.selection_slider.valueChanged.connect(self.update_display)
        self.left_button.clicked.connect(self.left_button_clicked)
        self.right_button.clicked.connect(self.right_button_clicked)

    def preload_data(self, item):
        if self.preloaded_intervals is None:
            self.preloaded_intervals = []
        series_id = item.text()
        series_events = self.events.loc[self.events["series_id"] == series_id]
        self.preloaded_intervals.clear()

        anglez, enmo, timestamp, extras = load_file(series_id)

        total_length = len(anglez)
        stride = total_length // (total_length // 2160)
        print(stride)

        # load every interval into memory
        k = 0
        while k + 17280 <= total_length:
            start = k
            end = k + 17280
            if total_length - (k + 17280) < stride:
                end = total_length
            interval_events = series_events.loc[(series_events["step"] >= start) & (series_events["step"] < end)]

            # load all the events into the interval
            events = []
            for j in range(len(interval_events)):
                event = interval_events.iloc[j]
                assert event["event"] in ["onset", "wakeup"]
                step = int(event["step"])
                if event["event"] == "onset":
                    events.append((timestamp.iloc[step], 1))
                    if step - 30 * 12 >= start:
                        events.append((timestamp.iloc[step - 30 * 12], 3))
                    if step + 30 * 12 < end:
                        events.append((timestamp.iloc[step + 30 * 12], 4))
                else:
                    events.append((timestamp.iloc[step], 2))
                    if step - 30 * 12 >= start:
                        events.append((timestamp.iloc[step - 30 * 12], 5))
                    if step + 30 * 12 < end:
                        events.append((timestamp.iloc[step + 30 * 12], 6))

            interval_anglez = anglez.iloc[start:end]
            interval_enmo = enmo.iloc[start:end]
            interval_timestamp = timestamp.iloc[start:end]
            local_extras = {}
            for key, value in extras.items():
                local_extras[key] = value[start:end]

            self.preloaded_intervals.append((series_id, start, end, interval_anglez, interval_enmo, local_extras, interval_timestamp, events))

            k += stride

        self.selection_slider.setValue(0)
        self.selection_slider.setMaximum(len(self.preloaded_intervals) - 1)

        # Set fixed width height
        min_anglez, max_anglez = anglez.min() / 35.52, anglez.max() / 35.52
        min_enmo, max_enmo = enmo.min() / 0.1018, enmo.max() / 0.1018
        min_y = min(min_anglez, min_enmo)
        max_y = max(max_anglez, max_enmo)
        length = max_y - min_y
        min_y -= length * 0.1
        max_y += length * 0.1

        self.display_widget.min_y = min_y
        self.display_widget.max_y = max_y

    def left_button_clicked(self):
        if self.preloaded_intervals is None:
            return

        self.selection_slider.setValue(max(self.selection_slider.value() - 1, 0))
        self.update_display()

    def right_button_clicked(self):
        if self.preloaded_intervals is None:
            return

        self.selection_slider.setValue(min(self.selection_slider.value() + 1, len(self.preloaded_intervals) - 1))
        self.update_display()

    def update_display(self):
        if self.preloaded_intervals is None:
            return

        selected_index = self.selection_slider.value()
        self.series_label.setText(str(selected_index))
        series_id, start, end, anglez, enmo, extras, timestamp, events = self.preloaded_intervals[selected_index]
        self.display_widget.plot_data(series_id, anglez, enmo, extras, timestamp, events)

def load_file(item):
    filename = "./individual_train_series/" + item + ".parquet"
    df = pd.read_parquet(filename)
    extras = {}
    folders = os.listdir(inference_regression_preds.FOLDER)
    for folder in folders:
        extras[folder] = np.load(os.path.join(inference_regression_preds.FOLDER, folder, item + ".npy"))

    """for folder in folders:
        extras[folder + "_kernel"] = generate_kernel_preds(extras[folder])"""

    folders2 = os.listdir(inference_confidence_preds.FOLDER)
    for folder in folders2:
        extras[folder + "_conf"] = np.load(os.path.join(inference_confidence_preds.FOLDER, folder, item + ".npy"))

    return df["anglez"], df["enmo"], df["timestamp"], extras

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


if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_widget = MainWidget()
    main_widget.show()
    sys.exit(app.exec_())
