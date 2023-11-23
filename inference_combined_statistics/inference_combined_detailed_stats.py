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
import metrics_ap
import convert_to_seriesid_events
import convert_to_pred_events

def plot_single_precision_recall_curve(ax, precisions, recalls, ap, proba, title):
    ax.plot(recalls, precisions)
    #ax.scatter(recalls, precisions, c=proba, cmap="coolwarm")
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, "AP: {:.4f}".format(ap), horizontalalignment="center", verticalalignment="center")

validation_AP_tolerances = [1, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30][::-1]
regression_labels_folders = [os.path.join("./inference_regression_statistics", "regression_labels", "Standard_5CV", "gaussian_kernel9"),
                             os.path.join("./inference_regression_statistics", "regression_labels", "Standard_5CV_Mid", "gaussian_kernel9"),
                             os.path.join("./inference_regression_statistics", "regression_labels", "Standard_5CV_Wide", "gaussian_kernel9")]
def validation_ap(fig: matplotlib.figure.Figure, predicted_events, gt_events, iou_probas_folder,
                  width, cutoff, augmentation, matrix_values_pruning,
                  probas_pruning, proba_pruning_radius, proba_pruning_inner_radius, proba_pruning_dropoff,
                  linear_dropoff, edges_probas_pruning):
    ap_onset_metrics = [metrics_ap.EventMetrics(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]
    ap_wakeup_metrics = [metrics_ap.EventMetrics(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]

    for series_id in all_series_ids:
        # load the regression predictions
        preds_locs = predicted_events[series_id]
        preds_locs_onset = preds_locs["onset"]
        preds_locs_wakeup = preds_locs["wakeup"]

        # prune using matrix values
        if matrix_values_pruning:
            matrix_values = np.load(os.path.join("./data_matrix_profile", "{}.npy".format(series_id)))
            preds_locs_onset = postprocessing.prune_matrix_profile(preds_locs_onset, matrix_values)
            preds_locs_wakeup = postprocessing.prune_matrix_profile(preds_locs_wakeup, matrix_values)

        # load the IOU probas
        onset_IOU_probas = np.load(os.path.join(iou_probas_folder, "{}_onset.npy".format(series_id)))
        wakeup_IOU_probas = np.load(os.path.join(iou_probas_folder, "{}_wakeup.npy".format(series_id)))
        original_length = len(onset_IOU_probas)

        # prune onset locs with wakeup IOU probas
        if edges_probas_pruning:
            if len(preds_locs_wakeup) > 0:
                preds_locs_wakeup = postprocessing.prune_wakeup_at_heads(preds_locs_wakeup, onset_IOU_probas)

        if probas_pruning:
            if len(preds_locs_onset) > 0:
                preds_locs_onset = postprocessing.prune_ROI(preds_locs_onset, onset_IOU_probas[preds_locs_onset],
                                                            proba_pruning_radius, proba_pruning_inner_radius, proba_pruning_dropoff)
            if len(preds_locs_wakeup) > 0:
                preds_locs_wakeup = postprocessing.prune_ROI(preds_locs_wakeup, wakeup_IOU_probas[preds_locs_wakeup],
                                                            proba_pruning_radius, proba_pruning_inner_radius, proba_pruning_dropoff)

        if augmentation:
            # load the kernel predictions
            onset_kernel_ensembled, wakeup_kernel_ensembled = None, None
            for k in range(len(regression_labels_folders)):
                onset_kernel = np.load(os.path.join(regression_labels_folders[k], "{}_onset.npy".format(series_id)))
                wakeup_kernel = np.load(os.path.join(regression_labels_folders[k], "{}_wakeup.npy".format(series_id)))
                if k == 0:
                    onset_kernel_ensembled = onset_kernel
                    wakeup_kernel_ensembled = wakeup_kernel
                else:
                    onset_kernel_ensembled = onset_kernel_ensembled + onset_kernel
                    wakeup_kernel_ensembled = wakeup_kernel_ensembled + wakeup_kernel
            onset_kernel_ensembled = onset_kernel_ensembled / len(regression_labels_folders)
            wakeup_kernel_ensembled = wakeup_kernel_ensembled / len(regression_labels_folders)

            # augment and restrict the probas
            preds_locs_onset, onset_IOU_probas = postprocessing.get_augmented_predictions(preds_locs_onset,
                                                            onset_kernel_ensembled, onset_IOU_probas, cutoff_thresh=cutoff)
            preds_locs_wakeup, wakeup_IOU_probas = postprocessing.get_augmented_predictions(preds_locs_wakeup,
                                                            wakeup_kernel_ensembled, wakeup_IOU_probas, cutoff_thresh=cutoff)
        else:
            # just restrict, without augmentation
            onset_IOU_probas = onset_IOU_probas[preds_locs_onset]
            wakeup_IOU_probas = wakeup_IOU_probas[preds_locs_wakeup]

        if linear_dropoff:
            # apply linear dropoff
            onset_IOU_probas = onset_IOU_probas * (1 - preds_locs_onset.astype(np.float32) / original_length)
            wakeup_IOU_probas = wakeup_IOU_probas * (1 - preds_locs_wakeup.astype(np.float32) / original_length)

        # get the ground truth
        gt_onset_locs = gt_events[series_id]["onset"]
        gt_wakeup_locs = gt_events[series_id]["wakeup"]

        # add info
        for ap_onset_metric, ap_wakeup_metric in zip(ap_onset_metrics, ap_wakeup_metrics):
            ap_onset_metric.add(pred_locs=preds_locs_onset, pred_probas=onset_IOU_probas, gt_locs=gt_onset_locs)
            ap_wakeup_metric.add(pred_locs=preds_locs_wakeup, pred_probas=wakeup_IOU_probas, gt_locs=gt_wakeup_locs)

    # compute average precision
    ap_onset_precisions, ap_onset_recalls, ap_onset_average_precisions, ap_onset_probas = [], [], [], []
    ap_wakeup_precisions, ap_wakeup_recalls, ap_wakeup_average_precisions, ap_wakeup_probas = [], [], [], []
    for ap_onset_metric, ap_wakeup_metric in zip(ap_onset_metrics, ap_wakeup_metrics):
        ap_onset_precision, ap_onset_recall, ap_onset_average_precision, ap_onset_proba = ap_onset_metric.get()
        ap_wakeup_precision, ap_wakeup_recall, ap_wakeup_average_precision, ap_wakeup_proba = ap_wakeup_metric.get()
        ap_onset_precisions.append(ap_onset_precision)
        ap_onset_recalls.append(ap_onset_recall)
        ap_onset_average_precisions.append(ap_onset_average_precision)
        ap_onset_probas.append(ap_onset_proba)
        ap_wakeup_precisions.append(ap_wakeup_precision)
        ap_wakeup_recalls.append(ap_wakeup_recall)
        ap_wakeup_average_precisions.append(ap_wakeup_average_precision)
        ap_wakeup_probas.append(ap_wakeup_proba)

    # draw the precision-recall curve using matplotlib onto file "epoch{}_AP.png".format(epoch) inside the ap_log_dir
    axes = fig.subplots(4, 5)
    fig.suptitle("Width: {}, Cutoff: {}, Augmentation: {} (Onset mAP: {}, Wakeup mAP: {})".format(width, cutoff, augmentation,
                                                                np.mean(ap_onset_average_precisions), np.mean(ap_wakeup_average_precisions)))
    for k in range(len(validation_AP_tolerances)):
        ax = axes[k // 5, k % 5]
        plot_single_precision_recall_curve(ax, ap_onset_precisions[k], ap_onset_recalls[k], ap_onset_average_precisions[k],
                                           ap_onset_probas[k], "Onset AP{}".format(validation_AP_tolerances[k]))
        ax = axes[(k + 10) // 5, (k + 10) % 5]
        plot_single_precision_recall_curve(ax, ap_wakeup_precisions[k], ap_wakeup_recalls[k], ap_wakeup_average_precisions[k],
                                           ap_wakeup_probas[k], "Wakeup AP{}".format(validation_AP_tolerances[k]))
    fig.subplots_adjust(wspace=0.5, hspace=0.7)

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

        # Create checkboxes
        self.checkbox_layout = QHBoxLayout()
        self.checkbox_augmentation = QCheckBox("Use Augmentation")
        self.checkbox_matrix_values_pruning = QCheckBox("Use Matrix Values Pruning")
        self.checkbox_probas_pruning = QCheckBox("Use Proba Pruning")
        self.checkbox_linear_dropoff = QCheckBox("Use Linear Dropoff")
        self.checkbox_edges_probas_pruning = QCheckBox("Use Edges Probas Pruning")
        self.checkbox_layout.addStretch(1)
        self.checkbox_layout.addWidget(self.checkbox_augmentation)
        self.checkbox_layout.addStretch(1)
        self.checkbox_layout.addWidget(self.checkbox_matrix_values_pruning)
        self.checkbox_layout.addStretch(1)
        self.checkbox_layout.addWidget(self.checkbox_probas_pruning)
        self.checkbox_layout.addStretch(1)
        self.checkbox_layout.addWidget(self.checkbox_linear_dropoff)
        self.checkbox_layout.addStretch(1)
        self.checkbox_layout.addWidget(self.checkbox_edges_probas_pruning)
        self.checkbox_layout.addStretch(1)
        self.main_layout.addLayout(self.checkbox_layout)

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
        self.slider_cutoff.setMaximum(1000)
        self.slider_cutoff.valueChanged.connect(self.update_cutoff_value)
        self.main_layout.addWidget(self.slider_cutoff)


        self.slider_pruning_radius_label = QLabel("Proba Pruning Radius: 0")
        self.slider_pruning_radius_label.setAlignment(Qt.AlignCenter)  # Centered the text for the slider label
        font_metrics = QFontMetrics(self.slider_pruning_radius_label.font())
        self.slider_pruning_radius_label.setFixedHeight(
            font_metrics.height())  # Set the height of the label to the height of the text
        self.main_layout.addWidget(self.slider_pruning_radius_label)

        self.slider_pruning_radius = QSlider(Qt.Horizontal)
        self.slider_pruning_radius.setMaximum(1440)
        self.slider_pruning_radius.valueChanged.connect(self.update_pruning_radius_value)
        self.main_layout.addWidget(self.slider_pruning_radius)

        self.slider_pruning_inner_radius_label = QLabel("Proba Pruning Inner Radius: 0")
        self.slider_pruning_inner_radius_label.setAlignment(Qt.AlignCenter)  # Centered the text for the slider label
        font_metrics = QFontMetrics(self.slider_pruning_inner_radius_label.font())
        self.slider_pruning_inner_radius_label.setFixedHeight(
            font_metrics.height())  # Set the height of the label to the height of the text
        self.main_layout.addWidget(self.slider_pruning_inner_radius_label)

        self.slider_pruning_inner_radius = QSlider(Qt.Horizontal)
        self.slider_pruning_inner_radius.setMaximum(360)
        self.slider_pruning_inner_radius.valueChanged.connect(self.update_pruning_inner_radius_value)
        self.main_layout.addWidget(self.slider_pruning_inner_radius)


        self.slider_pruning_dropoff_label = QLabel("Proba Pruning Dropoff: 0")
        self.slider_pruning_dropoff_label.setAlignment(Qt.AlignCenter)  # Centered the text for the slider label
        font_metrics = QFontMetrics(self.slider_pruning_dropoff_label.font())
        self.slider_pruning_dropoff_label.setFixedHeight(
            font_metrics.height())  # Set the height of the label to the height of the text
        self.main_layout.addWidget(self.slider_pruning_dropoff_label)

        self.slider_pruning_dropoff = QSlider(Qt.Horizontal)
        self.slider_pruning_dropoff.setMaximum(100)
        self.slider_pruning_dropoff.valueChanged.connect(self.update_pruning_dropoff_value)
        self.main_layout.addWidget(self.slider_pruning_dropoff)

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
        validation_ap(self.fig_plots, regression_predicted_events, per_series_id_events, selected_folder, self.get_union_width(), self.get_cutoff(),
                      self.checkbox_augmentation.isChecked(), self.checkbox_matrix_values_pruning.isChecked(),

                      self.checkbox_probas_pruning.isChecked(), self.get_pruning_radius(), self.get_pruning_inner_radius(), self.get_pruning_dropoff(),
                      self.checkbox_linear_dropoff.isChecked(), self.checkbox_edges_probas_pruning.isChecked())

        self.canvas_plots.draw()

    def get_union_width(self):
        return self.union_width_values[self.slider_union_width.value()]

    def get_cutoff(self):
        return self.slider_cutoff.value() / 1000.0

    def update_union_width_value(self, value):
        self.slider_union_width_label.setText("Union width: " + str(self.get_union_width()))
        #self.update_plots()

    def update_cutoff_value(self, value):
        self.slider_cutoff_label.setText("Cutoff: " + str(self.get_cutoff()))
        #self.update_plots()

    def get_pruning_radius(self):
        return self.slider_pruning_radius.value() * 12

    def update_pruning_radius_value(self, value):
        self.slider_pruning_radius_label.setText("Proba Pruning Radius: " + str(self.get_pruning_radius()))
        #self.update_plots()

    def get_pruning_inner_radius(self):
        return self.slider_pruning_inner_radius.value() * 12

    def update_pruning_inner_radius_value(self, value):
        self.slider_pruning_inner_radius_label.setText("Proba Pruning Inner Radius: " + str(self.get_pruning_inner_radius()))
        #self.update_plots()

    def get_pruning_dropoff(self):
        return self.slider_pruning_dropoff.value() / 100.0

    def update_pruning_dropoff_value(self, value):
        self.slider_pruning_dropoff_label.setText("Proba Pruning Dropoff: " + str(self.get_pruning_dropoff()))
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