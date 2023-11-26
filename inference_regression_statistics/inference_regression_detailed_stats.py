import sys
import os

from PySide2.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel, QCheckBox, QComboBox, QHBoxLayout, QPushButton
from PySide2.QtGui import QFontMetrics
from PySide2.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add root folder to sys.path
import postprocessing

def get_argmax_files_list(folder, is_onset):
    keyword = "onset" if is_onset else "wakeup"
    files_list = [os.path.join(folder, filename) for filename in os.listdir(folder) if (keyword in filename and "locmax" in filename)]
    kernel_vals_list = [os.path.join(folder, filename) for filename in os.listdir(folder) if (keyword in filename and "locmax" not in filename)]
    series_id_list = [filename.split("_")[0] for filename in os.listdir(folder) if (keyword in filename and "locmax" in filename)]
    return files_list, kernel_vals_list, series_id_list

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
    numbers = right_indices - left_indices

    for k in range(len(Y)):
        if left_indices[k] < right_indices[k]:
            values = X[left_indices[k]:right_indices[k]]
            differences = np.abs(values - Y[k])
            medians.append(np.median(differences))
        else:
            medians.append(n)

    return medians, numbers

def compute_numbers(X, Y, n=240):
    left_indices = np.searchsorted(X, Y - n, side="left")
    right_indices = np.searchsorted(X, Y + n, side="right")

    return right_indices - left_indices

def compute_metrics(selected_folder, is_onset, cutoff, pruning_radius, align_predictions: bool):
    files_list, kernel_vals_file_list, series_id_list = get_argmax_files_list(selected_folder, is_onset=is_onset)
    median_errs = []
    min_errs = []
    gaps = []
    numbers = []
    numbers_low = []
    for k in range(len(files_list)):
        argmax_data = np.load(files_list[k])
        event_locs = argmax_data[0, :].astype(np.int32)
        event_vals = argmax_data[1, :]

        event_locs = event_locs[event_vals > cutoff]  # restrict
        if (len(event_locs) > 0) and (pruning_radius > 0): # prune if needed
            event_locs = postprocessing.prune(event_locs, event_vals[event_vals > cutoff], pruning_radius=pruning_radius)
        if (len(event_locs) > 0) and align_predictions:
            all_kernel_values = np.load(kernel_vals_file_list[k])
            seconds_values = np.load("../data_naive/{}/secs.npy".format(series_id_list[k]))
            event_locs = postprocessing.align_predictions(event_locs, all_kernel_values, first_zero=
                                                          postprocessing.compute_first_zero(seconds_values))
            assert np.all(event_locs[1:] - event_locs[:-1] > 0), "event_locs must be strictly increasing"

        gt_events = loaded_events[series_id_list[k]]["onsets" if is_onset else "wakeups"]

        if len(gt_events) > 0:
            if len(event_locs) == 0:
                min_errs.extend([240] * len(gt_events))
                median_errs.extend([240] * len(gt_events))
            else:
                min_errs.extend(compute_mins(event_locs, gt_events))
                medians, series_numbers = compute_medians(event_locs, gt_events)
                median_errs.extend(medians)
                numbers.extend(series_numbers)
                numbers_low.extend(compute_numbers(event_locs, gt_events, n=120))

        if len(event_locs) > 1:
            gaps.extend(event_locs[1:] - event_locs[:-1])

    return min_errs, median_errs, gaps, numbers, numbers_low

def plot_graph_with_labels(ax, y_huber, y_laplace, y_gaussian, y_important_values=[12, 36, 60]):
    # Generate x-values
    x = np.arange(len(y_huber))

    # Plot the graphs
    ax.plot(x, y_huber, label="Huber")
    ax.plot(x, y_laplace, label="Laplace")
    ax.plot(x, y_gaussian, label="Gaussian")

    # Add rectangles for specific y-values
    ticks_vals = []
    for y_value in y_important_values:
        # Find the corresponding x-values for each graph
        y_idx = np.searchsorted(y_huber, y_value)
        if y_idx < len(y_huber):
            x_y1 = x[y_idx]
            y_y1 = y_huber[y_idx]
            rect_y1 = Rectangle((0, 0), x_y1, y_y1, fill=False, edgecolor="blue", linestyle=":")
            ax.add_patch(rect_y1)
            ticks_vals.append(x_y1)

        y_idx = np.searchsorted(y_laplace, y_value)
        if y_idx < len(y_laplace):
            x_y1 = x[y_idx]
            y_y1 = y_laplace[y_idx]
            rect_y1 = Rectangle((0, 0), x_y1, y_y1, fill=False, edgecolor="orange", linestyle=":")
            ax.add_patch(rect_y1)
            ticks_vals.append(x_y1)

        y_idx = np.searchsorted(y_gaussian, y_value)
        if y_idx < len(y_gaussian):
            x_y2 = x[y_idx]
            y_y2 = y_gaussian[y_idx]
            rect_y2 = Rectangle((0, 0), x_y2, y_y2, fill=False, edgecolor="green", linestyle=":")
            ax.add_patch(rect_y2)
            ticks_vals.append(x_y2)

    # Set x ticks
    ax.set_xticks(np.unique(list(np.arange(0, 101, 20)) + ticks_vals))


class MainWindow(QMainWindow):
    kernel_width_values = [2, 6, 9, 12, 36, 90, 360]

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
        self.fig_lowcount_distribution = Figure()
        self.fig_minerr_distribution_bounded = Figure()
        self.fig_medianerr_distribution_bounded = Figure()
        self.fig_count_distribution = Figure()

        # Create FigureCanvas objects
        self.canvas_minerr_distribution = FigureCanvas(self.fig_minerr_distribution)
        self.canvas_medianerr_distribution = FigureCanvas(self.fig_medianerr_distribution)
        self.canvas_lowcount_distribution = FigureCanvas(self.fig_lowcount_distribution)
        self.canvas_minerr_distribution_bounded = FigureCanvas(self.fig_minerr_distribution_bounded)
        self.canvas_medianerr_distribution_bounded = FigureCanvas(self.fig_medianerr_distribution_bounded)
        self.canvas_count_distribution = FigureCanvas(self.fig_count_distribution)

        # Create NavigationToolbars for each FigureCanvas
        self.toolbar_minerr_distribution = NavigationToolbar(self.canvas_minerr_distribution, self)
        self.toolbar_medianerr_distribution = NavigationToolbar(self.canvas_medianerr_distribution, self)
        self.toolbar_lowcount_distribution = NavigationToolbar(self.canvas_lowcount_distribution, self)
        self.toolbar_minerr_distribution_bounded = NavigationToolbar(self.canvas_minerr_distribution_bounded, self)
        self.toolbar_medianerr_distribution_bounded = NavigationToolbar(self.canvas_medianerr_distribution_bounded, self)
        self.toolbar_count_distribution = NavigationToolbar(self.canvas_count_distribution, self)

        # Create a horizontal layout for the plots
        self.plot_layout = QHBoxLayout()
        self.plot_left_widget = QWidget()
        self.plot_center_widget = QWidget()
        self.plot_right_widget = QWidget()
        self.plot_left_layout = QVBoxLayout(self.plot_left_widget)
        self.plot_center_layout = QVBoxLayout(self.plot_center_widget)
        self.plot_right_layout = QVBoxLayout(self.plot_right_widget)

        self.plot_left_layout.addWidget(self.toolbar_minerr_distribution)
        self.plot_left_layout.addWidget(self.canvas_minerr_distribution)
        self.plot_left_layout.addWidget(self.toolbar_minerr_distribution_bounded)
        self.plot_left_layout.addWidget(self.canvas_minerr_distribution_bounded)
        self.plot_center_layout.addWidget(self.toolbar_medianerr_distribution)
        self.plot_center_layout.addWidget(self.canvas_medianerr_distribution)
        self.plot_center_layout.addWidget(self.toolbar_medianerr_distribution_bounded)
        self.plot_center_layout.addWidget(self.canvas_medianerr_distribution_bounded)
        self.plot_right_layout.addWidget(self.toolbar_lowcount_distribution)
        self.plot_right_layout.addWidget(self.canvas_lowcount_distribution)
        self.plot_right_layout.addWidget(self.toolbar_count_distribution)
        self.plot_right_layout.addWidget(self.canvas_count_distribution)

        self.plot_layout.addWidget(self.plot_left_widget)
        self.plot_layout.addWidget(self.plot_center_widget)
        self.plot_layout.addWidget(self.plot_right_widget)

        # Add plot layout to the main layout
        self.main_layout.addLayout(self.plot_layout)

        # Create a dropdown menu
        self.dropdown = QComboBox()
        self.dropdown.addItems(list(self.folders.keys()))  # Added items to the dropdown menu
        self.main_layout.addWidget(self.dropdown)

        # Create checkboxes
        self.checkbox_layout = QHBoxLayout()
        self.checkbox_aligned = QCheckBox("Use Aligned Predictions")
        self.checkbox_onset = QCheckBox("Use Onset (otherwise Wakeup)")
        self.checkbox_layout.addStretch(1)
        self.checkbox_layout.addWidget(self.checkbox_aligned)
        self.checkbox_layout.addStretch(1)
        self.checkbox_layout.addWidget(self.checkbox_onset)
        self.checkbox_layout.addStretch(1)
        self.main_layout.addLayout(self.checkbox_layout)

        # Create sliders
        self.slider_kernel_width_label = QLabel("Kernel width: {}".format(self.kernel_width_values[0]))
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

        self.slider_pruning_label = QLabel("Pruning: 0")
        self.slider_pruning_label.setAlignment(Qt.AlignCenter)  # Centered the text for the slider label
        font_metrics = QFontMetrics(self.slider_pruning_label.font())
        self.slider_pruning_label.setFixedHeight(
            font_metrics.height())  # Set the height of the label to the height of the text
        self.main_layout.addWidget(self.slider_pruning_label)

        self.slider_pruning = QSlider(Qt.Horizontal)
        self.slider_pruning.setMaximum(30)
        self.slider_pruning.valueChanged.connect(self.update_pruning_value)
        self.main_layout.addWidget(self.slider_pruning)

        # Create a "Plot" button
        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.update_plots)
        self.main_layout.addWidget(self.plot_button)

    def update_plots(self):
        selected_summary_name = self.get_selected_summary_name()
        selected_folder_huber = self.get_selected_folder(kernel_shape="huber")
        selected_folder_laplace = self.get_selected_folder(kernel_shape="laplace")
        selected_folder_gaussian = self.get_selected_folder(kernel_shape="gaussian")
        min_errs_huber, median_errs_huber, gaps_huber, numbers_huber, numbers_huber_low =\
            compute_metrics(selected_folder_huber, is_onset=self.checkbox_onset.isChecked(), cutoff=self.get_cutoff(),
                            pruning_radius=self.get_pruning(), align_predictions=self.checkbox_aligned.isChecked())
        min_errs_laplace, median_errs_laplace, gaps_laplace, numbers_laplace, numbers_laplace_low = \
            compute_metrics(selected_folder_laplace, is_onset=self.checkbox_onset.isChecked(), cutoff=self.get_cutoff(),
                            pruning_radius=self.get_pruning(), align_predictions=self.checkbox_aligned.isChecked())
        min_errs_gaussian, median_errs_gaussian, gaps_gaussian, numbers_gaussian, numbers_gaussian_low =\
            compute_metrics(selected_folder_gaussian, is_onset=self.checkbox_onset.isChecked(), cutoff=self.get_cutoff(),
                            pruning_radius=self.get_pruning(), align_predictions=self.checkbox_aligned.isChecked())

        # Plot the distributions, with x-axis being quantiles (0-100), and y-axis the value of the metric at that quantile
        self.fig_minerr_distribution.clear()
        self.fig_medianerr_distribution.clear()
        self.fig_lowcount_distribution.clear()
        self.fig_minerr_distribution_bounded.clear()
        self.fig_medianerr_distribution_bounded.clear()
        self.fig_count_distribution.clear()

        ax_minerr = self.fig_minerr_distribution.add_subplot(111)
        ax_medianerr = self.fig_medianerr_distribution.add_subplot(111)
        ax_lowcount = self.fig_lowcount_distribution.add_subplot(111)
        ax_minerr_bounded = self.fig_minerr_distribution_bounded.add_subplot(111)
        ax_medianerr_bounded = self.fig_medianerr_distribution_bounded.add_subplot(111)
        ax_count = self.fig_count_distribution.add_subplot(111)

        ax_minerr.plot(np.arange(101), np.percentile(min_errs_huber, np.arange(101)), label="Huber")
        ax_minerr.plot(np.arange(101), np.percentile(min_errs_laplace, np.arange(101)), label="Laplace")
        ax_minerr.plot(np.arange(101), np.percentile(min_errs_gaussian, np.arange(101)), label="Gaussian")
        ax_medianerr.plot(np.arange(101), np.percentile(median_errs_huber, np.arange(101)), label="Huber")
        ax_medianerr.plot(np.arange(101), np.percentile(median_errs_laplace, np.arange(101)), label="Laplace")
        ax_medianerr.plot(np.arange(101), np.percentile(median_errs_gaussian, np.arange(101)), label="Gaussian")
        ax_lowcount.plot(np.arange(101), np.percentile(numbers_huber_low, np.arange(101)), label="Huber")
        ax_lowcount.plot(np.arange(101), np.percentile(numbers_laplace_low, np.arange(101)), label="Laplace")
        ax_lowcount.plot(np.arange(101), np.percentile(numbers_gaussian_low, np.arange(101)), label="Gaussian")
        plot_graph_with_labels(ax_minerr_bounded, np.percentile(min_errs_huber, np.arange(101)),
                               np.percentile(min_errs_laplace, np.arange(101)),
                               np.percentile(min_errs_gaussian, np.arange(101)))
        plot_graph_with_labels(ax_medianerr_bounded, np.percentile(median_errs_huber, np.arange(101)),
                               np.percentile(median_errs_laplace, np.arange(101)),
                               np.percentile(median_errs_gaussian, np.arange(101)))
        ax_minerr_bounded.set_ylim([0, 80])
        ax_medianerr_bounded.set_ylim([0, 80])
        ax_minerr_bounded.set_yticks(np.arange(0, 81, 20))
        ax_medianerr_bounded.set_yticks(np.arange(0, 81, 20))
        ax_count.plot(np.arange(101), np.percentile(numbers_huber, np.arange(101)), label="Huber")
        ax_count.plot(np.arange(101), np.percentile(numbers_laplace, np.arange(101)), label="Laplace")
        ax_count.plot(np.arange(101), np.percentile(numbers_gaussian, np.arange(101)), label="Gaussian")

        ax_minerr.set_title("Min error ({})".format(selected_summary_name))
        ax_medianerr.set_title("Median error ({})".format(selected_summary_name))
        ax_lowcount.set_title("W=120 Neighborhood Count ({})".format(selected_summary_name))
        ax_minerr_bounded.set_title("Min error ({})".format(selected_summary_name))
        ax_medianerr_bounded.set_title("Median error ({})".format(selected_summary_name))
        ax_count.set_title("W=240 Neighborhood Count ({})".format(selected_summary_name))

        ax_minerr.legend()
        ax_medianerr.legend()
        ax_lowcount.legend()
        ax_minerr_bounded.legend()
        ax_medianerr_bounded.legend()
        ax_count.legend()

        self.canvas_minerr_distribution.draw()
        self.canvas_medianerr_distribution.draw()
        self.canvas_lowcount_distribution.draw()
        self.canvas_minerr_distribution_bounded.draw()
        self.canvas_medianerr_distribution_bounded.draw()
        self.canvas_count_distribution.draw()

    def get_kernel_width(self):
        return self.kernel_width_values[self.slider_kernel_width.value()]

    def get_cutoff(self):
        return self.slider_cutoff.value() / 100.0

    def get_pruning(self):
        return self.slider_pruning.value() * 12

    def get_selected_summary_name(self):
        results_summary_folder = self.folders[self.dropdown.currentText()]
        with open(os.path.join(results_summary_folder, "name.txt"), "r") as f:
            return f.read().strip()

    def get_selected_folder(self, kernel_shape):
        results_summary_folder = self.folders[self.dropdown.currentText()]
        if "gaussian_kernel" in os.listdir(results_summary_folder):
            # learnable sigmas. this means the kernel shape must be huber, and the kernel width is learnt
            return os.path.join(results_summary_folder, "{}_kernel".format(kernel_shape))

        kernel_width = self.get_kernel_width()

        return os.path.join(results_summary_folder, "{}_kernel{}".format(kernel_shape, kernel_width))

    def update_kernel_width_value(self, value):
        self.slider_kernel_width_label.setText("Kernel width: " + str(self.get_kernel_width()))
        #self.update_plots()

    def update_cutoff_value(self, value):
        self.slider_cutoff_label.setText("Cutoff: " + str(self.get_cutoff()))
        #self.update_plots()

    def update_pruning_value(self, value):
        self.slider_pruning_label.setText("Pruning: " + str(self.get_pruning()))
        #self.update_plots()

if __name__ == "__main__":
    events = pd.read_csv("../data/train_events.csv")
    events = events.dropna()
    loaded_events = {}
    all_series_ids = [filename.split(".")[0] for filename in os.listdir("../individual_train_series")]
    for series_id in all_series_ids:
        series_events = events[events["series_id"] == series_id]
        if len(series_events) > 0:
            series_onsets = np.unique(series_events.loc[series_events["event"] == "onset"]["step"].to_numpy())
            series_wakeups = np.unique(series_events.loc[series_events["event"] == "wakeup"]["step"].to_numpy())

            loaded_events[series_id] = {
                "onsets": series_onsets,
                "wakeups": series_wakeups
            }
        else:
            loaded_events[series_id] = {
                "onsets": np.array([]),
                "wakeups": np.array([])
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