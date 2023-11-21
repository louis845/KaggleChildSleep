import sys
import os
import json

import matplotlib.figure
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
import convert_to_seriesid_events
import convert_to_pred_events

def plot_length_stats(fig: matplotlib.figure.Figure, gt_events, iou_probas_folder, width,
                      cutoff):

    onset_nopreds = 0
    wakeup_nopreds = 0
    all_onset_distances = []
    all_wakeup_distances = []

    for series_id in all_series_ids:
        # load the IOU probas
        onset_IOU_probas = np.load(os.path.join(iou_probas_folder, "{}_onset.npy".format(series_id)))
        wakeup_IOU_probas = np.load(os.path.join(iou_probas_folder, "{}_wakeup.npy".format(series_id)))

        # get the ground truth events
        onset_gt = np.array(gt_events[series_id]["onset"], dtype=np.int32)
        wakeup_gt = np.array(gt_events[series_id]["wakeup"], dtype=np.int32)

        # compute the distances
        if np.all(onset_IOU_probas <= cutoff):
            onset_nopreds += len(onset_gt)
        else:
            onset_distances = postprocessing.compute_distances(onset_IOU_probas, cutoff, onset_gt)
            all_onset_distances.extend(onset_distances)
        if np.all(wakeup_IOU_probas <= cutoff):
            wakeup_nopreds += len(wakeup_gt)
        else:
            wakeup_distances = postprocessing.compute_distances(wakeup_IOU_probas, cutoff, wakeup_gt)
            all_wakeup_distances.extend(wakeup_distances)


    # draw the precision-recall curve using matplotlib onto file "epoch{}_AP.png".format(epoch) inside the ap_log_dir
    axes = fig.subplots(5, 2)
    fig.suptitle("Length plots. Width: {}".format(width))

    # plot the cumulative distribution of distances, left plot is onset, right plot is wakeup
    # x-axis should be percentile (0-100), and y-axis is the distance

    # onset
    onset_distances = np.array(all_onset_distances)
    axes[0, 0].plot(np.linspace(0, 1, 1001), np.percentile(onset_distances, np.linspace(0, 100, 1001)))
    axes[0, 0].set_xlabel("Percentile")
    axes[0, 0].set_ylabel("Distance (Step)")
    axes[0, 0].set_title("Onset (No Pred: {})".format(onset_nopreds))

    onset_distances = np.array(all_onset_distances)
    axes[1, 0].plot(np.linspace(0, 0.99, 991), np.percentile(onset_distances, np.linspace(0, 99, 991)))
    axes[1, 0].set_xlabel("Percentile")
    axes[1, 0].set_ylabel("Distance (Step)")
    axes[1, 0].set_title("Onset (No Pred: {})".format(onset_nopreds))

    onset_distances = np.array(all_onset_distances)
    axes[2, 0].plot(np.linspace(0, 0.98, 981), np.percentile(onset_distances, np.linspace(0, 98, 981)))
    axes[2, 0].set_xlabel("Percentile")
    axes[2, 0].set_ylabel("Distance (Step)")
    axes[2, 0].set_title("Onset (No Pred: {})".format(onset_nopreds))

    onset_distances = np.array(all_onset_distances)
    axes[3, 0].plot(np.linspace(0, 0.97, 971), np.percentile(onset_distances, np.linspace(0, 97, 971)))
    axes[3, 0].set_xlabel("Percentile")
    axes[3, 0].set_ylabel("Distance (Step)")
    axes[3, 0].set_title("Onset (No Pred: {})".format(onset_nopreds))

    onset_distances = np.array(all_onset_distances)
    axes[4, 0].plot(np.linspace(0, 0.95, 951), np.percentile(onset_distances, np.linspace(0, 95, 951)))
    axes[4, 0].set_xlabel("Percentile")
    axes[4, 0].set_ylabel("Distance (Step)")
    axes[4, 0].set_title("Onset (No Pred: {})".format(onset_nopreds))

    # wakeup
    wakeup_distances = np.array(all_wakeup_distances)
    axes[0, 1].plot(np.linspace(0, 1, 1001), np.percentile(wakeup_distances, np.linspace(0, 100, 1001)))
    axes[0, 1].set_xlabel("Percentile")
    axes[0, 1].set_ylabel("Distance (Step)")
    axes[0, 1].set_title("Wakeup (No Pred: {})".format(wakeup_nopreds))

    wakeup_distances = np.array(all_wakeup_distances)
    axes[1, 1].plot(np.linspace(0, 0.99, 991), np.percentile(wakeup_distances, np.linspace(0, 99, 991)))
    axes[1, 1].set_xlabel("Percentile")
    axes[1, 1].set_ylabel("Distance (Step)")
    axes[1, 1].set_title("Wakeup (No Pred: {})".format(wakeup_nopreds))

    wakeup_distances = np.array(all_wakeup_distances)
    axes[2, 1].plot(np.linspace(0, 0.98, 981), np.percentile(wakeup_distances, np.linspace(0, 98, 981)))
    axes[2, 1].set_xlabel("Percentile")
    axes[2, 1].set_ylabel("Distance (Step)")
    axes[2, 1].set_title("Wakeup (No Pred: {})".format(wakeup_nopreds))

    wakeup_distances = np.array(all_wakeup_distances)
    axes[3, 1].plot(np.linspace(0, 0.97, 971), np.percentile(wakeup_distances, np.linspace(0, 97, 971)))
    axes[3, 1].set_xlabel("Percentile")
    axes[3, 1].set_ylabel("Distance (Step)")
    axes[3, 1].set_title("Wakeup (No Pred: {})".format(wakeup_nopreds))

    wakeup_distances = np.array(all_wakeup_distances)
    axes[4, 1].plot(np.linspace(0, 0.95, 951), np.percentile(wakeup_distances, np.linspace(0, 95, 951)))
    axes[4, 1].set_xlabel("Percentile")
    axes[4, 1].set_ylabel("Distance (Step)")
    axes[4, 1].set_title("Wakeup (No Pred: {})".format(wakeup_nopreds))

class MainWindow(QMainWindow):
    union_width_values = [31, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 120]

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        with open(os.path.join("./inference_combined_statistics/inference_combined_preds_options.json"), "r") as f:
            self.options = json.load(f)

        # Initialize folders
        self.folders = {option["name"]: option for option in self.options}

        self.setWindowTitle("Detailed combined statistics")

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        self.main_layout = QVBoxLayout(self.main_widget)

        # Create matplotlib plots
        self.fig_plots = Figure()

        # Create FigureCanvas objects
        self.canvas_plots = FigureCanvas(self.fig_plots)

        # Create NavigationToolbars for each FigureCanvas
        self.toolbar_plots = NavigationToolbar(self.canvas_plots, self)

        # Create a horizontal layout for the plots
        self.plot_layout = QVBoxLayout()
        self.plot_layout.addWidget(self.toolbar_plots)
        self.plot_layout.addWidget(self.canvas_plots)

        # Add plot layout to the main layout
        self.main_layout.addLayout(self.plot_layout)

        # Create a dropdown menu
        self.dropdown = QComboBox()
        self.dropdown.addItems(list(self.folders.keys()))  # Added items to the dropdown menu
        self.main_layout.addWidget(self.dropdown)

        # Create sliders
        self.slider_union_width_label = QLabel("Union width: {}".format(self.union_width_values[0]))
        self.slider_union_width_label.setAlignment(Qt.AlignCenter)  # Centered the text for the slider label
        font_metrics = QFontMetrics(self.slider_union_width_label.font())
        self.slider_union_width_label.setFixedHeight(font_metrics.height())  # Set the height of the label to the height of the text
        self.main_layout.addWidget(self.slider_union_width_label)

        self.slider_union_width = QSlider(Qt.Horizontal)
        self.slider_union_width.setMinimum(0)
        self.slider_union_width.setMaximum(len(self.union_width_values) - 1)
        self.slider_union_width.valueChanged.connect(self.update_union_width_value)
        self.main_layout.addWidget(self.slider_union_width)

        self.slider_cutoff_label = QLabel("Cutoff: 0")
        self.slider_cutoff_label.setAlignment(Qt.AlignCenter)  # Centered the text for the slider label
        font_metrics = QFontMetrics(self.slider_cutoff_label.font())
        self.slider_cutoff_label.setFixedHeight(
            font_metrics.height())  # Set the height of the label to the height of the text
        self.main_layout.addWidget(self.slider_cutoff_label)

        self.slider_cutoff = QSlider(Qt.Horizontal)
        self.slider_cutoff.setMaximum(100)
        self.slider_cutoff.valueChanged.connect(self.update_cutoff_value)
        self.main_layout.addWidget(self.slider_cutoff)

        # Create a "Plot" button
        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.update_plots)
        self.main_layout.addWidget(self.plot_button)

    def get_selected_folder(self):
        union_width = self.get_union_width()
        option_name = self.dropdown.currentText()
        selected_folder = os.path.join("./inference_combined_statistics/combined_predictions", self.folders[option_name]["out_folder"], "width{}".format(union_width))
        return selected_folder

    def update_plots(self):
        self.fig_plots.clear()

        selected_folder = self.get_selected_folder()
        plot_length_stats(self.fig_plots, per_series_id_events, selected_folder, self.get_union_width(), self.get_cutoff())

        self.canvas_plots.draw()

    def get_union_width(self):
        return self.union_width_values[self.slider_union_width.value()]

    def get_cutoff(self):
        return self.slider_cutoff.value() / 100.0

    def update_union_width_value(self, value):
        self.slider_union_width_label.setText("Union width: " + str(self.get_union_width()))
        #self.update_plots()

    def update_cutoff_value(self, value):
        self.slider_cutoff_label.setText("Cutoff: " + str(self.get_cutoff()))
        #self.update_plots()

if __name__ == "__main__":
    all_series_ids = [filename.split(".")[0] for filename in os.listdir("./individual_train_series")]

    per_series_id_events = convert_to_seriesid_events.get_events_per_seriesid()
    regression_predicted_events = convert_to_pred_events.load_all_pred_events_into_dict()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    screen = app.screens()[0]
    screen_geometry = screen.geometry()
    window.move((screen_geometry.width() - window.width()) / 2, (screen_geometry.height() - window.height()) / 2)
    sys.exit(app.exec_())