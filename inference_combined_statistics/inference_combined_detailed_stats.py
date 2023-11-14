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
import metrics_ap

class MainWindow(QMainWindow):
    union_width_values = [31, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 120]

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Initialize folders
        #self.folders = {subfolder: os.path.join("./regression_labels", subfolder) for subfolder in os.listdir("./regression_labels")}

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
        #self.dropdown.addItems(list(self.folders.keys()))  # Added items to the dropdown menu
        self.main_layout.addWidget(self.dropdown)

        # Create checkboxes
        self.checkbox_layout = QHBoxLayout()
        self.checkbox_augmentation = QCheckBox("Use Augmentation")
        self.checkbox_layout.addStretch(1)
        self.checkbox_layout.addWidget(self.checkbox_augmentation)
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
        self.slider_cutoff.setMaximum(100)
        self.slider_cutoff.valueChanged.connect(self.update_cutoff_value)
        self.main_layout.addWidget(self.slider_cutoff)

        # Create a "Plot" button
        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.update_plots)
        self.main_layout.addWidget(self.plot_button)

    def update_plots(self):
        pass

    def get_union_width(self):
        return self.union_width_values[self.slider_union_width.value()]

    def get_cutoff(self):
        return self.slider_cutoff.value() / 100.0

    def update_union_width_value(self, value):
        self.slider_union_width_label.setText("Union width: " + str(self.get_union_width()))
        self.update_plots()

    def update_cutoff_value(self, value):
        self.slider_cutoff_label.setText("Cutoff: " + str(self.get_cutoff()))
        self.update_plots()

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