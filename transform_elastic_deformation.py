import sys
import os
import pandas as pd
import numpy as np
from PySide2.QtWidgets import QApplication, QVBoxLayout, QFileDialog, QWidget, QHBoxLayout, QPushButton, QListWidget, QTabWidget, QListWidgetItem, QSplitter, QSlider, QLabel, QCheckBox, QComboBox
from PySide2.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.interpolate import interp1d

def generate_deformation_indices(length):
    assert length % 12 == 0, "Length must be divisible by 12 (multiple of 1min)"
    def_type = np.random.randint(0, 4)
    if def_type == 0:
        deformed_time_indices = np.random.randint(0, length, size=length)
        deformed_time_indices = np.sort(deformed_time_indices)
    else:
        if def_type == 1:
            strength = 2
        elif def_type == 2:
            strength = 3
        else:
            strength = 6

        deformed_time_indices = np.random.randint(0, length // strength, size=length // strength) * strength
        deformed_time_indices = np.sort(deformed_time_indices)
        ground = np.arange(length // strength)
        deformed_time_indices = interp1d(ground, deformed_time_indices, kind="cubic")(np.arange(length))
        deformed_time_indices = np.clip(np.round(deformed_time_indices).astype(np.int32), 0, length - 1)

    return deformed_time_indices

def deform_time_series(time_series, deformed_time_indices):
    assert time_series.shape[1] == len(deformed_time_indices)
    assert len(time_series.shape) == 2

    deformed_time_series = time_series[:, deformed_time_indices]
    return deformed_time_series

def deform_v_time_series(time_series, deform_scale=True):
    z_def_type = np.random.randint(0, 3)
    if z_def_type > 0:
        if z_def_type == 1:
            z_deformation = np.random.randint(-90, 91, size=91 + 90) + np.random.rand(91 + 90)
            ground = np.arange(start=-90, stop=91, dtype=np.float32) / 35.5195
        else:
            z_deformation = (np.random.randint(-30, 31, size=31 + 30) + np.random.rand(31 + 30)) * 3
            ground = np.arange(start=-30, stop=31, dtype=np.float32) * 3 / 35.5195

        z_deformation = np.sort(z_deformation) / 35.52
        time_series = interp1d(ground, z_deformation, kind="cubic")(time_series)

    if deform_scale:
        if np.random.rand() > 0.5:
            scale = 1.0 + 0.1 * np.random.rand() * np.sin(10 * (0.5 + np.random.rand()) * np.linspace(0, 2 * np.pi, num=len(time_series)) + np.random.rand() * 2 * np.pi)
            time_series = time_series * scale

    return time_series

def find_closest_index(x, val):
    idx = np.searchsorted(x, val, side="left")

    if idx > 0 and (idx == len(x) or np.fabs(val - x[idx - 1]) <= np.fabs(val - x[idx])):
        return idx - 1
    else:
        return idx

if __name__ == "__main__":
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


        def plot_data(self, title, anglez, enmo, timestamp, events, start, deform=False):
            self.axis.clear()
            x = pd.to_datetime(timestamp)  # Automatically parses the timestamp
            y1 = anglez.to_numpy(dtype=np.float32) / 35.52 # std computed by check_series_properties.py
            y2 = enmo.to_numpy(dtype=np.float32) / 0.1018 # std computed by check_series_properties.py

            if deform:
                deformed_time_indices = generate_deformation_indices(len(x))

                deformed_y = deform_time_series(np.stack([y1, y2], axis=0), deformed_time_indices)

                y1 = deformed_y[0, :]
                y1 = deform_v_time_series(y1)
                y2 = deformed_y[1, :]

            self.axis.set_ylim([self.min_y, self.max_y])
            self.axis.plot(x, y1, label="anglez")
            self.axis.plot(x, y2, label="enmo")

            for event_time, event_step, event_type in events:
                color = "blue" if event_type == 1 else "red"
                if deform:
                    event_step = event_step - start
                    event_step = find_closest_index(deformed_time_indices, event_step)
                    self.axis.axvline(x.iloc[event_step], color=color, alpha=0.5, linestyle="--")
                else:
                    self.axis.axvline(pd.to_datetime(event_time), color=color, alpha=0.5, linestyle="--")

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
            self.display_widget_deformed = MatplotlibWidget()

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

            self.refresh_button = QPushButton("Refresh")

            self.main_layout.addWidget(self.display_widget)
            self.main_layout.addWidget(self.display_widget_deformed)
            self.main_layout.addWidget(self.labels_widget)
            self.main_layout.addWidget(self.selection_slider)
            self.main_layout.addWidget(self.refresh_button)

            # Add series ids
            series_ids = [x[:-8] for x in os.listdir("individual_train_series")]
            series_ids.sort()
            for series_id in series_ids:
                self.series_id_items.addItem(series_id)

            self.series_id_items.itemDoubleClicked.connect(self.preload_data)
            self.selection_slider.valueChanged.connect(self.update_display)
            self.left_button.clicked.connect(self.left_button_clicked)
            self.right_button.clicked.connect(self.right_button_clicked)
            self.refresh_button.clicked.connect(self.update_display)

        def preload_data(self, item):
            if self.preloaded_intervals is None:
                self.preloaded_intervals = []
            series_id = item.text()
            series_events = self.events.loc[self.events["series_id"] == series_id]
            self.preloaded_intervals.clear()

            anglez, enmo, timestamp = load_file(series_id)

            total_length = len(anglez)
            stride = total_length // (total_length // 2160)

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
                        events.append((timestamp.iloc[step], step, 1))
                    else:
                        events.append((timestamp.iloc[step], step, 2))

                interval_anglez = anglez.iloc[start:end]
                interval_enmo = enmo.iloc[start:end]
                interval_timestamp = timestamp.iloc[start:end]

                self.preloaded_intervals.append(
                    (series_id, start, end, interval_anglez, interval_enmo, interval_timestamp, events))

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
            self.display_widget_deformed.min_y = min_y
            self.display_widget_deformed.max_y = max_y

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
            series_id, start, end, anglez, enmo, timestamp, events = self.preloaded_intervals[selected_index]
            self.display_widget.plot_data(series_id, anglez, enmo, timestamp, events, start)
            self.display_widget_deformed.plot_data(series_id + " (Deformed)", anglez, enmo, timestamp, events, start, deform=True)

    def load_file(item):
        filename = "./individual_train_series/" + item + ".parquet"
        df = pd.read_parquet(filename)

        return df["anglez"], df["enmo"], df["timestamp"]

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

    app = QApplication(sys.argv)

    main_widget = MainWidget()
    main_widget.show()
    sys.exit(app.exec_())
