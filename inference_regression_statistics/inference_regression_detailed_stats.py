import sys
import os

from PySide2.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel, QCheckBox, QComboBox, QHBoxLayout, QPushButton
from PySide2.QtGui import QFontMetrics
from PySide2.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
import numpy as np

def get_argmax_files_list(folder, is_onset):
    keyword = "onset" if is_onset else "wakeup"
    files_list = [os.path.join(folder, filename) for filename in os.listdir(folder) if (keyword in filename and "locmax" in filename)]
    series_id_list = [filename.split("_")[0] for filename in os.listdir(folder) if (keyword in filename and "locmax" in filename)]
    return files_list, series_id_list

def find_closest_indices(X, Y):
    # X should be sorted, and X, Y should be 1D arrays
    indices = np.searchsorted(X, Y)
    indices = np.clip(indices, 1, len(X) - 1)
    left = X[indices - 1]
    right = X[indices]
    indices -= Y - left < right - Y

    return indices

def compute_mins(X, Y):
    closest_indices = find_closest_indices(X, Y)
    return np.abs(X[closest_indices] - Y)

def compute_medians(X, Y, n=240):
    medians = []
    left_indices = np.searchsorted(X, Y - n, side="left")
    right_indices = np.searchsorted(X, Y + n, side="right")

    for k in range(len(Y)):
        if left_indices[k] < right_indices[k]:
            values = X[left_indices[k]:right_indices[k]]
            differences = np.abs(values - Y[k])
            medians.append(np.median(differences))
        else:
            medians.append(n)

    return medians

def compute_numbers(X, Y, n=240):
    left_indices = np.searchsorted(X, Y - n, side="left")
    right_indices = np.searchsorted(X, Y + n, side="right")

    return right_indices - left_indices
    

class MainWindow(QMainWindow):
    kernel_width_values = [2, 4, 6, 9, 12, 24, 36, 48, 60, 90, 120, 180, 240, 360]

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Initialize folders
        self.folders = {subfolder: os.path.join("./regression_labels", subfolder) for subfolder in os.listdir("./regression_labels")}

        self.setWindowTitle("Detailed regression statistics")

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        self.main_layout = QVBoxLayout(self.main_widget)

        # Create matplotlib plots
        self.fig_minerr_distribution = Figure()
        self.fig_medianerr_distribution = Figure()
        self.fig_gap_distribution = Figure()

        # Create FigureCanvas objects
        self.canvas_minerr_distribution = FigureCanvas(self.fig_minerr_distribution)
        self.canvas_medianerr_distribution = FigureCanvas(self.fig_medianerr_distribution)
        self.canvas_gap_distribution = FigureCanvas(self.fig_gap_distribution)

        # Create NavigationToolbars for each FigureCanvas
        self.toolbar_minerr_distribution = NavigationToolbar(self.canvas_minerr_distribution, self)
        self.toolbar_medianerr_distribution = NavigationToolbar(self.canvas_medianerr_distribution, self)
        self.toolbar_gap_distribution = NavigationToolbar(self.canvas_gap_distribution, self)

        # Create a horizontal layout for the plots
        self.plot_layout = QHBoxLayout()
        self.plot_layout.addWidget(self.canvas_minerr_distribution)
        self.plot_layout.addWidget(self.toolbar_minerr_distribution)
        self.plot_layout.addWidget(self.canvas_medianerr_distribution)
        self.plot_layout.addWidget(self.toolbar_medianerr_distribution)
        self.plot_layout.addWidget(self.canvas_gap_distribution)
        self.plot_layout.addWidget(self.toolbar_gap_distribution)

        # Add plot layout to the main layout
        self.main_layout.addLayout(self.plot_layout)

        # Create a dropdown menu
        self.dropdown = QComboBox()
        self.dropdown.addItems(list(self.folders.keys()))  # Added items to the dropdown menu
        self.main_layout.addWidget(self.dropdown)

        # Create checkboxes
        self.checkbox_layout = QHBoxLayout()
        self.checkbox_huber = QCheckBox("Use Huber Kernel (otherwise Gaussian kernel)")
        self.checkbox_onset = QCheckBox("Use Onset (otherwise Wakeup)")
        self.checkbox_layout.addStretch(1)
        self.checkbox_layout.addWidget(self.checkbox_huber)
        self.checkbox_layout.addStretch(1)
        self.checkbox_layout.addWidget(self.checkbox_onset)
        self.checkbox_layout.addStretch(1)
        self.main_layout.addLayout(self.checkbox_layout)

        # Create sliders
        self.slider_kernel_width_label = QLabel("Kernel width: 0")
        self.slider_kernel_width_label.setAlignment(Qt.AlignCenter)  # Centered the text for the slider label
        font_metrics = QFontMetrics(self.slider_kernel_width_label.font())
        self.slider_kernel_width_label.setFixedHeight(font_metrics.height())  # Set the height of the label to the height of the text
        self.main_layout.addWidget(self.slider_kernel_width_label)

        self.slider_kernel_width = QSlider(Qt.Horizontal)
        self.slider_kernel_width.setMinimum(0)
        self.slider_kernel_width.setMaximum(len(self.kernel_width_values) - 1)
        self.slider_kernel_width.valueChanged.connect(self.update_kernel_width_value)
        self.main_layout.addWidget(self.slider_kernel_width)

        self.slider_cutoff_label = QLabel("Cutoff: 0")
        self.slider_cutoff_label.setAlignment(Qt.AlignCenter)  # Centered the text for the slider label
        font_metrics = QFontMetrics(self.slider_cutoff_label.font())
        self.slider_cutoff_label.setFixedHeight(
            font_metrics.height())  # Set the height of the label to the height of the text
        self.main_layout.addWidget(self.slider_cutoff_label)

        self.slider_cutoff = QSlider(Qt.Horizontal)
        self.slider_cutoff.setMaximum(1000)
        self.slider_cutoff.valueChanged.connect(self.update_cutoff_value)
        self.main_layout.addWidget(self.slider_cutoff)

        # Create a "Plot" button
        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.update_plots)
        self.main_layout.addWidget(self.plot_button)

    def update_plots(self):
        selected_folder = self.get_selected_folder()
        selected_summary_name = self.get_selected_summary_name()

        # Load and compute the metrics
        files_list, series_id_list = get_argmax_files_list(selected_folder, is_onset=self.checkbox_onset.isChecked())
        median_errs = []
        min_errs = []
        gaps = []
        for k in range(len(files_list)):
            argmax_data = np.load(files_list[k])
            event_locs = argmax_data[0, :].astype(np.int32)
            event_vals = argmax_data[1, :]

            event_locs = event_locs[event_vals > self.slider_cutoff.value()] # restrict

            gt_events = loaded_events[series_id_list[k]]["onsets" if self.checkbox_onset.isChecked() else "wakeup"]

            if len(event_locs) == 0:
                min_errs.extend([240] * len(gt_events))
                median_errs.extend([240] * len(gt_events))
            else:
                min_errs.extend(compute_mins(event_locs, gt_events))
                median_errs.extend(compute_medians(event_locs, gt_events))

            if len(event_locs) > 1:
                gaps.extend(event_locs[1:] - event_locs[:-1])

        # Plot the distributions, with x-axis being quantiles (0-100), and y-axis the value of the metric at that quantile
        self.fig_minerr_distribution.clear()
        self.fig_medianerr_distribution.clear()
        self.fig_gap_distribution.clear()

        ax_minerr = self.fig_minerr_distribution.add_subplot(111)
        ax_medianerr = self.fig_medianerr_distribution.add_subplot(111)
        ax_gap = self.fig_gap_distribution.add_subplot(111)

        ax_minerr.plot(np.arange(101), np.percentile(min_errs, np.arange(101)))
        ax_medianerr.plot(np.arange(101), np.percentile(median_errs, np.arange(101)))
        ax_gap.plot(np.arange(101), np.percentile(gaps, np.arange(101)))

        ax_minerr.set_title("Min error ({})".format(selected_summary_name))
        ax_medianerr.set_title("Median error ({})".format(selected_summary_name))
        ax_gap.set_title("Gap ({})".format(selected_summary_name))

        self.canvas_minerr_distribution.draw()
        self.canvas_medianerr_distribution.draw()
        self.canvas_gap_distribution.draw()


    def get_kernel_width(self):
        return self.kernel_width_values[self.slider_kernel_width.value()]

    def get_selected_summary_name(self):
        results_summary_folder = self.folders[self.dropdown.currentText()]
        with open(os.path.join(results_summary_folder, "name.txt"), "r") as f:
            return f.read().strip()

    def get_selected_folder(self):
        kernel_width = self.get_kernel_width()
        kernel_shape = "huber" if self.checkbox_huber.isChecked() else "gaussian"
        results_summary_folder = self.folders[self.dropdown.currentText()]

        return os.path.join(results_summary_folder, "{}_kernel{}".format(kernel_shape, kernel_width))

    def update_kernel_width_value(self, value):
        self.slider_kernel_width_label.setText("Kernel width: " + str(self.get_kernel_width()))
        self.update_plots()

    def update_cutoff_value(self, value):
        self.slider_cutoff_label.setText("Cutoff: " + str(value))
        self.update_plots()

if __name__ == "__main__":
    events = pd.read_csv("../data/train_events.csv")
    events = events.dropna()
    loaded_events = {}
    for series_id in events["series_id"].unique():
        series_events = events[events["series_id"] == series_id]
        series_onsets = np.unique(series_events.loc[series_events["event"] == "onset"]["step"].to_numpy())
        series_wakeups = np.unique(series_events.loc[series_events["event"] == "wakeup"]["step"].to_numpy())

        loaded_events[series_id] = {
            "onsets": series_onsets,
            "wakeups": series_wakeups
        }

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    screen = app.screens()[0]
    screen_geometry = screen.geometry()
    window.move((screen_geometry.width() - window.width()) / 2, (screen_geometry.height() - window.height()) / 2)
    sys.exit(app.exec_())