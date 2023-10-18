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

import manager_folds
import convert_to_h5py_naive
import convert_to_good_events
import convert_to_interval_events


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


    def plot_data(self, title, anglez, enmo, onsets, wakeups):
        self.axis.clear()
        x = np.arange(len(anglez))
        self.axis.plot(x, anglez, label="anglez")
        self.axis.plot(x, enmo, label="enmo")
        self.axis.plot(x, onsets, label="onsets")
        self.axis.plot(x, wakeups, label="wakeups")

        self.axis.set_title(title)
        self.axis.legend()
        self.canvas.draw()

class MainWidget(QWidget):
    def __init__(self, sampler: convert_to_interval_events.SemiSyntheticIntervalEventsSampler):
        super(MainWidget, self).__init__(None)
        self.setWindowTitle("Visualization of time series intervals and events")
        self.resize(1280, 720)

        # Create layout
        self.layout = QVBoxLayout(self)

        self.main_widget = QWidget()
        self.layout.addWidget(self.main_widget)

        self.main_layout = QVBoxLayout(self.main_widget)

        self.display_widget = MatplotlibWidget()
        self.refresh_button = QPushButton("Refresh")

        self.main_layout.addWidget(self.display_widget)
        self.main_layout.addWidget(self.refresh_button)

        self.refresh_button.clicked.connect(self.refresh_button_clicked)

        self.sampler = sampler
        sampler.shuffle()

    def update_display(self):
        if self.preloaded_intervals is None:
            return

        selected_index = self.selection_slider.value()
        self.series_label.setText(str(selected_index))
        series_id, start, end, anglez, enmo, timestamp, events = self.preloaded_intervals[selected_index]
        self.display_widget.plot_data(series_id, anglez, enmo, timestamp, events)

    def refresh_button_clicked(self):
        if self.sampler.entries_remaining() == 0:
            self.sampler.shuffle()

        accel_datas, event_segmentations, _ = self.sampler.sample(1, 0)
        accel_data = accel_datas[0, ...]
        event_segmentation = event_segmentations[0, ...]

        anglez = accel_data[0, :]
        enmo = accel_data[1, :]
        event_onsets = np.repeat(event_segmentation[0, :], 12)
        event_wakeup = np.repeat(event_segmentation[1, :], 12)

        self.display_widget.plot_data("Sampled series", anglez, enmo, event_onsets, event_wakeup)


if __name__ == "__main__":
    all_data = convert_to_h5py_naive.load_all_data_into_dict()
    all_segmentations = convert_to_interval_events.load_all_segmentations()

    cutmix_length = 540
    cutmix_skip = 540
    entries = manager_folds.load_dataset("fold_1_train")

    sampler = convert_to_interval_events.SemiSyntheticIntervalEventsSampler(entries, all_data,
                                                                             all_segmentations,
                                                                             spliced_good_events=convert_to_good_events.GoodEventsSplicedSampler(
                                                                                 data_dict=convert_to_good_events.load_all_data_into_dict(),
                                                                                 entries_sublist=entries,
                                                                                 expected_length=cutmix_length
                                                                             ),
                                                                             cutmix_skip=cutmix_skip)

    app = QApplication(sys.argv)

    main_widget = MainWidget(sampler)
    main_widget.show()
    sys.exit(app.exec_())
