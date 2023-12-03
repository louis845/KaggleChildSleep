import sys
import os

import pandas as pd
import numpy as np
from PySide2.QtWidgets import (QWidget, QVBoxLayout, QComboBox, QTableWidget,
                               QTableWidgetItem, QHeaderView, QScrollArea, QApplication, QMainWindow, QHBoxLayout)
from PySide2.QtCore import Qt


class ModelMetricsVisualizerWidget(QWidget):
    def __init__(self, models):
        super(ModelMetricsVisualizerWidget, self).__init__()

        self.loaded_metrics = None
        self.models = models

        self.init_ui()

    def init_ui(self):
        # Set the layout
        layout = QVBoxLayout()

        # Create the table widget and put it inside a scroll area
        self.table_widget = QTableWidget()
        self.table_widget.setEditTriggers(QTableWidget.NoEditTriggers)  # Make the table read-only
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.table_widget)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        # Create the dropdown menu
        self.model_selector = QComboBox()
        self.model_selector.addItems(self.models)
        self.model_selector.currentIndexChanged.connect(self.load_metrics)
        layout.addWidget(self.model_selector)

        # Set the layout to the widget
        self.setLayout(layout)
        self.set_etched_border()

    def set_etched_border(self):
        self.setStyleSheet("QWidget { border: 2px groove gray; }")

    def load_metrics(self):
        model = self.model_selector.currentText()
        if model:
            try:
                file_path = "./models/{}/val_metrics.csv".format(model)
                loaded_metrics = pd.read_csv(file_path, index_col=0)
                loaded_metrics["epoch"] = np.arange(1, loaded_metrics.shape[0] + 1)
                loaded_metrics["val_mAP"] = (loaded_metrics["val_onset_dense_loc_softmax_mAP"] +
                                                loaded_metrics["val_wakeup_dense_loc_softmax_mAP"]) / 2
                loaded_metrics = loaded_metrics[["epoch", "val_mAP",
                                                 "val_onset_dense_loc_softmax_mAP", "val_wakeup_dense_loc_softmax_mAP",
                                                 "val_loss", "val_class_loss", "val_entropy_loss",
                                                 "val_class_metric_tpr", "val_class_metric_fpr"]]
                loaded_metrics = loaded_metrics.sort_values(by=["val_mAP"], ascending=False)
                self.loaded_metrics = loaded_metrics
                self.update_metrics_table(self.loaded_metrics)
            except Exception as e:
                print(f"An error occurred while loading metrics: {e}")

    def update_metrics_table(self, table):
        self.table_widget.setRowCount(table.shape[0])
        self.table_widget.setColumnCount(table.shape[1])
        self.table_widget.setHorizontalHeaderLabels(table.columns)

        # Populate the table
        for row in range(table.shape[0]):
            for col in range(table.shape[1]):
                value = table.iloc[row, col]
                formatted_value = "{:.5g}".format(value) if isinstance(value, float) else str(value)
                self.table_widget.setItem(row, col, QTableWidgetItem(formatted_value))

        # Hide index column
        self.table_widget.verticalHeader().setVisible(False)

        # Resize columns to fit content
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)


class ModelMetricsVisualizer(QMainWindow):
    def __init__(self, models, num_displays):
        super(ModelMetricsVisualizer, self).__init__()

        self.models = models
        self.num_displays = num_displays
        self.model_visualizers = []

        self.init_ui()

    def init_ui(self):
        # Create the main widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create a QHBoxLayout which will contain the visualizers
        self.horizontal_layout = QHBoxLayout()

        # Create a scroll area to hold the horizontal layout
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Add visualizers to the layout
        for _ in range(self.num_displays):
            visualizer = ModelMetricsVisualizerWidget(self.models)
            self.model_visualizers.append(visualizer)
            self.horizontal_layout.addWidget(visualizer)

        # Create a container widget and set the horizontal layout
        self.container_widget = QWidget()
        self.container_widget.setLayout(self.horizontal_layout)

        # Set the container widget as the scroll area's widget
        self.scroll_area.setWidget(self.container_widget)

        # Set the scroll area as the central widget of the window
        self.central_widget.layout = QVBoxLayout()
        self.central_widget.layout.addWidget(self.scroll_area)
        self.central_widget.setLayout(self.central_widget.layout)

        # Set window title
        self.setWindowTitle("Model Metrics Visualizer")

if __name__ == "__main__":
    os.listdir("./models")
    models = [model for model in os.listdir("./models") if
                (os.path.isdir(os.path.join("./models", model)) and ("density" in model or "event10fold" in model))
             ]
    models.sort()

    app = QApplication(sys.argv)
    visualizer = ModelMetricsVisualizer(models, 2)
    visualizer.resize(1080, 720)
    visualizer.show()
    visualizer.raise_()
    visualizer.activateWindow()
    screens = app.screens()[0]
    screen_geometry = screens.geometry()
    visualizer.move((screen_geometry.width() - visualizer.width()) / 2, (screen_geometry.height() - visualizer.height()) / 2)
    sys.exit(app.exec_())