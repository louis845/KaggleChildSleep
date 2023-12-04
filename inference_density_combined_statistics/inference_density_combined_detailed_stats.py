import sys
import os
import json
from typing import Iterator

import matplotlib.figure
from PySide2.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel, QCheckBox, QHBoxLayout, QPushButton, QSplitter, QComboBox, QScrollArea
from PySide2.QtGui import QFontMetrics
from PySide2.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import h5py
import tqdm

import postprocessing
import metrics_ap
import convert_to_seriesid_events
import manager_folds
import model_event_density_unet
import bad_series_list

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

def get_regression_preds_locs(selected_regression_folders: list[str], series_id: str, alignment=True):
    # regression settings
    cutoff = 0.01
    pruning = 72

    # compute locs here
    num_regression_preds = 0
    onset_kernel_vals, wakeup_kernel_vals = None, None
    for folder in selected_regression_folders:
        onset_kernel = np.load(os.path.join(folder, "{}_onset.npy".format(series_id)))
        wakeup_kernel = np.load(os.path.join(folder, "{}_wakeup.npy".format(series_id)))
        if onset_kernel_vals is None:
            onset_kernel_vals = onset_kernel
            wakeup_kernel_vals = wakeup_kernel
        else:
            onset_kernel_vals = onset_kernel_vals + onset_kernel
            wakeup_kernel_vals = wakeup_kernel_vals + wakeup_kernel
        num_regression_preds += 1

    onset_kernel_vals = onset_kernel_vals / num_regression_preds
    wakeup_kernel_vals = wakeup_kernel_vals / num_regression_preds

    onset_locs = (onset_kernel_vals[1:-1] > onset_kernel_vals[0:-2]) & (onset_kernel_vals[1:-1] > onset_kernel_vals[2:])
    onset_locs = np.argwhere(onset_locs).flatten() + 1
    wakeup_locs = (wakeup_kernel_vals[1:-1] > wakeup_kernel_vals[0:-2]) & (wakeup_kernel_vals[1:-1] > wakeup_kernel_vals[2:])
    wakeup_locs = np.argwhere(wakeup_locs).flatten() + 1

    onset_values = onset_kernel_vals[onset_locs]
    wakeup_values = wakeup_kernel_vals[wakeup_locs]

    onset_locs = onset_locs[onset_values > cutoff]
    wakeup_locs = wakeup_locs[wakeup_values > cutoff]

    if pruning > 0:
        if len(onset_locs) > 0:
            onset_locs = postprocessing.prune(onset_locs, onset_values[onset_values > cutoff], pruning)
        if len(wakeup_locs) > 0:
            wakeup_locs = postprocessing.prune(wakeup_locs, wakeup_values[wakeup_values > cutoff], pruning)

    if alignment:
        seconds_values = np.load("./data_naive/{}/secs.npy".format(series_id))
        first_zero = postprocessing.compute_first_zero(seconds_values)
        if len(onset_locs) > 0:
            onset_locs = postprocessing.align_predictions(onset_locs, onset_kernel_vals, first_zero=first_zero)
        if len(wakeup_locs) > 0:
            wakeup_locs = postprocessing.align_predictions(wakeup_locs, wakeup_kernel_vals, first_zero=first_zero)

    return onset_locs, wakeup_locs, onset_kernel_vals, wakeup_kernel_vals

def event_density_file_logit_iterator(selected_density_folder, series_id) -> Iterator[dict[str, np.ndarray]]:
    # load the logits
    logits_file = os.path.join(selected_density_folder, series_id, "intervals.h5")
    with h5py.File(logits_file, "r") as f:
        intervals_start = f["intervals_start"][:]
        intervals_end = f["intervals_end"][:]
        intervals_logits = f["intervals_logits"][:]
        intervals_event_presence = f["intervals_event_presence"][:]

    for k in range(len(intervals_start)):
        interval_start = intervals_start[k]
        interval_end = intervals_end[k]
        interval_logits = intervals_logits[k]
        interval_event_presence = intervals_event_presence[k]
        yield {
            "interval_start": interval_start,
            "interval_end": interval_end,
            "interval_logits": interval_logits,
            "interval_event_presence": interval_event_presence
        }

def validation_ap(fig: matplotlib.figure.Figure, gt_events,
                  selected_density_folders: list[str], selected_regression_folders: list[str],
                  cutoff, augmentation, augmentation_cutoff, matrix_values_pruning,
                  linear_dropoff, postalignment,

                  selected_fold: int,

                  exclude_bad_segmentations: bool):
    selected_series_ids = subfolds[selected_fold]

    ap_onset_metrics = [metrics_ap.EventMetrics(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]
    ap_wakeup_metrics = [metrics_ap.EventMetrics(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]

    cutoff_onset_values = 0
    cutoff_wakeup_values = 0
    total_onset_values = 0
    total_wakeup_values = 0

    for series_id in tqdm.tqdm(selected_series_ids):
        if exclude_bad_segmentations and series_id in bad_series_list.noisy_bad_segmentations:
            continue

        # compute the regression predictions
        preds_locs_onset, preds_locs_wakeup, onset_kernel_vals, wakeup_kernel_vals = get_regression_preds_locs(selected_regression_folders, series_id,
                                                                                                               alignment=not postalignment)
        total_length = len(onset_kernel_vals)

        # load and compute the probas
        onset_locs_all_probas, wakeup_locs_all_probas = None, None
        for k in range(len(selected_density_folders)):
            logit_loader = event_density_file_logit_iterator(selected_density_folders[k], series_id)
            _, onset_locs_probas, wakeup_locs_probas = model_event_density_unet.event_density_probas_from_interval_info(
                                                                             interval_info_stream=logit_loader,
                                                                             total_length=total_length,
                                                                             predicted_locations=[{
                                                                                    "onset": preds_locs_onset,
                                                                                    "wakeup": preds_locs_wakeup
                                                                             }],
                                                                             return_probas=False)
            if onset_locs_all_probas is None:
                onset_locs_all_probas = onset_locs_probas[0]
                wakeup_locs_all_probas = wakeup_locs_probas[0]
            else:
                onset_locs_all_probas += onset_locs_probas[0]
                wakeup_locs_all_probas += wakeup_locs_probas[0]
        onset_locs_all_probas /= len(selected_density_folders)
        wakeup_locs_all_probas /= len(selected_density_folders)

        # prune using cutoff. also compute cutoff stats
        total_onset_values += len(preds_locs_onset)
        total_wakeup_values += len(preds_locs_wakeup)
        if cutoff > 0:
            cutoff_onset_values += np.sum(onset_locs_all_probas > cutoff)
            cutoff_wakeup_values += np.sum(wakeup_locs_all_probas > cutoff)

            preds_locs_onset = preds_locs_onset[onset_locs_all_probas > cutoff]
            onset_locs_all_probas = onset_locs_all_probas[onset_locs_all_probas > cutoff]
            preds_locs_wakeup = preds_locs_wakeup[wakeup_locs_all_probas > cutoff]
            wakeup_locs_all_probas = wakeup_locs_all_probas[wakeup_locs_all_probas > cutoff]

        if postalignment:
            seconds_values = np.load("./data_naive/{}/secs.npy".format(series_id))
            first_zero = postprocessing.compute_first_zero(seconds_values)
            if len(preds_locs_onset) > 0:
                original_length = len(preds_locs_onset)
                preds_locs_onset = postprocessing.align_predictions(preds_locs_onset, onset_kernel_vals, first_zero=first_zero)
                assert len(preds_locs_onset) == original_length
            if len(preds_locs_wakeup) > 0:
                original_length = len(preds_locs_wakeup)
                preds_locs_wakeup = postprocessing.align_predictions(preds_locs_wakeup, wakeup_kernel_vals, first_zero=first_zero)
                assert len(preds_locs_wakeup) == original_length

        if augmentation:
            # augment and restrict the probas
            preds_locs_onset, onset_locs_all_probas = postprocessing.get_augmented_predictions_density(preds_locs_onset,
                                                            onset_kernel_vals, onset_locs_all_probas, cutoff_thresh=augmentation_cutoff)
            preds_locs_wakeup, wakeup_locs_all_probas = postprocessing.get_augmented_predictions_density(preds_locs_wakeup,
                                                            wakeup_kernel_vals, wakeup_locs_all_probas, cutoff_thresh=augmentation_cutoff)

        # prune using matrix values
        if matrix_values_pruning:
            matrix_values = np.load(os.path.join("./data_matrix_profile", "{}.npy".format(series_id)))
            preds_locs_onset_restrict = postprocessing.prune_matrix_profile(preds_locs_onset, matrix_values,
                                                                            return_idx=True)
            preds_locs_wakeup_restrict = postprocessing.prune_matrix_profile(preds_locs_wakeup, matrix_values,
                                                                             return_idx=True)

            preds_locs_onset = preds_locs_onset[preds_locs_onset_restrict]
            onset_locs_all_probas = onset_locs_all_probas[preds_locs_onset_restrict]
            preds_locs_wakeup = preds_locs_wakeup[preds_locs_wakeup_restrict]
            wakeup_locs_all_probas = wakeup_locs_all_probas[preds_locs_wakeup_restrict]

        if linear_dropoff:
            # apply linear dropoff
            onset_locs_all_probas = onset_locs_all_probas * (1 - preds_locs_onset.astype(np.float32) / total_length)
            wakeup_locs_all_probas = wakeup_locs_all_probas * (1 - preds_locs_wakeup.astype(np.float32) / total_length)

        # get the ground truth
        gt_onset_locs = gt_events[series_id]["onset"]
        gt_wakeup_locs = gt_events[series_id]["wakeup"]

        # exclude end if excluding bad segmentations
        if exclude_bad_segmentations and series_id in bad_series_list.bad_segmentations_tail:
            if len(gt_onset_locs) > 0 and len(gt_wakeup_locs) > 0:
                last = gt_wakeup_locs[-1] + 8640
                onset_cutoff = np.searchsorted(preds_locs_onset, last, side="right")
                wakeup_cutoff = np.searchsorted(preds_locs_wakeup, last, side="right")

                preds_locs_onset = preds_locs_onset[:onset_cutoff]
                preds_locs_wakeup = preds_locs_wakeup[:wakeup_cutoff]
                onset_locs_all_probas = onset_locs_all_probas[:onset_cutoff]
                wakeup_locs_all_probas = wakeup_locs_all_probas[:wakeup_cutoff]

        # add info
        for ap_onset_metric, ap_wakeup_metric in zip(ap_onset_metrics, ap_wakeup_metrics):
            ap_onset_metric.add(pred_locs=preds_locs_onset, pred_probas=onset_locs_all_probas, gt_locs=gt_onset_locs)
            ap_wakeup_metric.add(pred_locs=preds_locs_wakeup, pred_probas=wakeup_locs_all_probas, gt_locs=gt_wakeup_locs)

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


    axes = fig.subplots(4, 5)
    fig.suptitle("Cutoff: {}, Augmentation: {} (Onset mAP: {}, Wakeup mAP: {})".format(cutoff, augmentation,
                                                                np.mean(ap_onset_average_precisions), np.mean(ap_wakeup_average_precisions)))
    for k in range(len(validation_AP_tolerances)):
        ax = axes[k // 5, k % 5]
        plot_single_precision_recall_curve(ax, ap_onset_precisions[k], ap_onset_recalls[k], ap_onset_average_precisions[k],
                                           ap_onset_probas[k], "Onset AP{}".format(validation_AP_tolerances[k]))
        ax = axes[(k + 10) // 5, (k + 10) % 5]
        plot_single_precision_recall_curve(ax, ap_wakeup_precisions[k], ap_wakeup_recalls[k], ap_wakeup_average_precisions[k],
                                           ap_wakeup_probas[k], "Wakeup AP{}".format(validation_AP_tolerances[k]))
    fig.subplots_adjust(wspace=0.5, hspace=0.7)

    print("Total onset values: {}      Remaining onset values: {}".format(total_onset_values, cutoff_onset_values))
    print("Total wakeup values: {}      Remaining wakeup values: {}".format(total_wakeup_values, cutoff_wakeup_values))
    print("Onset cutoff ratio: {:.4f}".format(cutoff_onset_values / total_onset_values))
    print("Wakeup cutoff ratio: {:.4f}".format(cutoff_wakeup_values / total_wakeup_values))

class MainWindow(QMainWindow):
    union_width_values = [31, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 120]

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        with open(os.path.join("./inference_density_statistics/inference_density_preds_options.json"), "r") as f:
            self.options = json.load(f)
        with open(os.path.join("./inference_regression_statistics/inference_regression_preds_options.json"), "r") as f:
            self.regression_options = json.load(f)

        # Initialize folders
        self.folders = {option["name"]: option for option in self.options}
        for regress_opt in self.regression_options:
            regress_opt["out_folder"] = regress_opt["name"].replace(" ", "_").replace("(", "").replace(")", "")
        self.regression_folders = {option["name"]: option for option in self.regression_options}

        # Initialize UI
        self.setWindowTitle("Detailed combined statistics")

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        self.main_layout = QVBoxLayout(self.main_widget)
        self.top_bottom_splitter = QSplitter(Qt.Vertical)
        self.main_layout.addWidget(self.top_bottom_splitter)

        self.top_widget = QWidget(self.top_bottom_splitter)
        self.top_bottom_splitter.addWidget(self.top_widget)
        self.top_layout = QVBoxLayout(self.top_widget)

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
        self.top_layout.addLayout(self.plot_layout)

        # Add a dropdown menu with options "Huber" "Gaussian" "Laplace"
        self.dropdown_kernel_shape = QComboBox()
        self.dropdown_kernel_shape.addItems(["Huber", "Gaussian", "Laplace"])
        self.top_layout.addWidget(self.dropdown_kernel_shape)

        self.dropdown_fold = QComboBox()
        self.dropdown_fold.addItems(["All"] + ["Fold {}".format(k) for k in range(1, 6)])
        self.top_layout.addWidget(self.dropdown_fold)

        # Create checkboxes
        self.checkbox_layout = QHBoxLayout()
        self.checkbox_augmentation = QCheckBox("Use Augmentation")
        self.checkbox_matrix_values_pruning = QCheckBox("Use Matrix Values Pruning")
        self.checkbox_linear_dropoff = QCheckBox("Use Linear Dropoff")
        self.checkbox_postalignment = QCheckBox("Use Post Alignment")
        self.checkbox_exclude_bad = QCheckBox("Exclude Bad Segmentations")
        self.checkbox_layout.addStretch(1)
        self.checkbox_layout.addWidget(self.checkbox_augmentation)
        self.checkbox_layout.addStretch(1)
        self.checkbox_layout.addWidget(self.checkbox_matrix_values_pruning)
        self.checkbox_layout.addStretch(1)
        self.checkbox_layout.addWidget(self.checkbox_linear_dropoff)
        self.checkbox_layout.addStretch(1)
        self.checkbox_layout.addWidget(self.checkbox_postalignment)
        self.checkbox_layout.addStretch(1)
        self.checkbox_layout.addWidget(self.checkbox_exclude_bad)
        self.checkbox_layout.addStretch(1)
        self.top_layout.addLayout(self.checkbox_layout)

        # Create sliders
        self.slider_cutoff_label = QLabel("Cutoff: 0")
        self.slider_cutoff_label.setAlignment(Qt.AlignCenter)  # Centered the text for the slider label
        font_metrics = QFontMetrics(self.slider_cutoff_label.font())
        self.slider_cutoff_label.setFixedHeight(
            font_metrics.height())  # Set the height of the label to the height of the text
        self.top_layout.addWidget(self.slider_cutoff_label)

        self.slider_cutoff = QSlider(Qt.Horizontal)
        self.slider_cutoff.setMaximum(1000)
        self.slider_cutoff.valueChanged.connect(self.update_cutoff_value)
        self.top_layout.addWidget(self.slider_cutoff)

        self.slider_aug_cutoff_label = QLabel("Aug Cutoff: 0")
        self.slider_aug_cutoff_label.setAlignment(Qt.AlignCenter)  # Centered the text for the slider label
        font_metrics = QFontMetrics(self.slider_aug_cutoff_label.font())
        self.slider_aug_cutoff_label.setFixedHeight(
            font_metrics.height())  # Set the height of the label to the height of the text
        self.top_layout.addWidget(self.slider_aug_cutoff_label)

        self.slider_aug_cutoff = QSlider(Qt.Horizontal)
        self.slider_aug_cutoff.setMaximum(1000)
        self.slider_aug_cutoff.valueChanged.connect(self.update_aug_cutoff_value)
        self.top_layout.addWidget(self.slider_aug_cutoff)

        # Create two list of checkboxes
        self.bottom_scroll_area = QScrollArea()
        self.top_bottom_splitter.addWidget(self.bottom_scroll_area)
        self.bottom_widget = QWidget(self.bottom_scroll_area)
        self.bottom_scroll_area.setWidget(self.bottom_widget)
        self.bottom_layout = QHBoxLayout(self.bottom_widget)
        self.bottom_scroll_area.setWidgetResizable(True)

        self.left_checkbox_layout = QVBoxLayout()
        self.right_checkbox_layout = QVBoxLayout()
        self.bottom_layout.addLayout(self.left_checkbox_layout)
        self.bottom_layout.addLayout(self.right_checkbox_layout)

        self.density_choice_checkboxes = []
        for option_name in self.folders:
            checkbox = QCheckBox(option_name)
            checkbox.setChecked(False)
            self.left_checkbox_layout.addWidget(checkbox)
            self.density_choice_checkboxes.append(checkbox)

        self.regression_choice_checkboxes = []
        for option_name in self.regression_folders:
            checkbox = QCheckBox(option_name)
            checkbox.setChecked(False)
            self.right_checkbox_layout.addWidget(checkbox)
            self.regression_choice_checkboxes.append(checkbox)

        # Create a "Plot" button
        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.update_plots)
        self.main_layout.addWidget(self.plot_button)

    def get_selected_folders(self):
        selected_kernel_shape = self.dropdown_kernel_shape.currentText().lower()
        selected_density_folders = []
        selected_regression_folders = []

        for checkbox in self.density_choice_checkboxes:
            if checkbox.isChecked():
                folder_name = self.folders[checkbox.text()]["folder_name"]
                folder = os.path.join("./inference_density_statistics/density_labels", folder_name)
                assert os.path.isdir(folder)
                selected_density_folders.append(folder)

        for checkbox in self.regression_choice_checkboxes:
            if checkbox.isChecked():
                folder_name = self.regression_folders[checkbox.text()]["out_folder"]
                folder = os.path.join("./inference_regression_statistics/regression_labels", folder_name)
                assert os.path.isdir(folder)

                if os.path.isdir(os.path.join(folder, "{}_kernel".format(selected_kernel_shape))):
                    folder = os.path.join(folder, "{}_kernel".format(selected_kernel_shape))
                else:
                    folder = os.path.join(folder, "{}_kernel9".format(selected_kernel_shape))
                assert os.path.isdir(folder)
                selected_regression_folders.append(folder)

        return selected_density_folders, selected_regression_folders

    def update_plots(self):
        selected_density_folders, selected_regression_folders = self.get_selected_folders()
        selected_fold = self.dropdown_fold.currentIndex()
        if len(selected_density_folders) == 0 or len(selected_regression_folders) == 0:
            return

        self.fig_plots.clear()
        validation_ap(self.fig_plots, per_series_id_events, selected_density_folders, selected_regression_folders, self.get_cutoff(),
                      self.checkbox_augmentation.isChecked(), self.get_aug_cutoff(),
                      self.checkbox_matrix_values_pruning.isChecked(),

                      self.checkbox_linear_dropoff.isChecked(), self.checkbox_postalignment.isChecked(),

                      selected_fold=selected_fold,

                      exclude_bad_segmentations=self.checkbox_exclude_bad.isChecked())
        self.canvas_plots.draw()

    def get_cutoff(self):
        return self.slider_cutoff.value() / 1000.0

    def update_cutoff_value(self, value):
        self.slider_cutoff_label.setText("Cutoff: " + str(self.get_cutoff()))

    def get_aug_cutoff(self):
        return self.slider_aug_cutoff.value() / 100.0

    def update_aug_cutoff_value(self, value):
        self.slider_aug_cutoff_label.setText("Aug Cutoff: " + str(self.get_aug_cutoff()))

def load_subfold(k):
    return manager_folds.load_dataset("fold_{}_val_5cv".format(k))

if __name__ == "__main__":
    all_series_ids = [filename.split(".")[0] for filename in os.listdir("./individual_train_series")]
    subfolds = [all_series_ids] + [load_subfold(k) for k in range(1, 6)]

    per_series_id_events = convert_to_seriesid_events.get_events_per_seriesid()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    screen = app.screens()[0]
    screen_geometry = screen.geometry()
    window.move((screen_geometry.width() - window.width()) / 2, (screen_geometry.height() - window.height()) / 2)
    sys.exit(app.exec_())