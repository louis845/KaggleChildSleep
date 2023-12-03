import sys
import os

import pandas as pd
from PySide2.QtWidgets import (QWidget, QVBoxLayout, QComboBox, QTableWidget,
                               QTableWidgetItem, QHeaderView, QScrollArea, QApplication, QMainWindow, QHBoxLayout)
from PySide2.QtCore import Qt
from PySide2.QtGui import QPalette, QColor


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
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor('white'))
        self.setPalette(palette)
        self.setFrameStyle(QWidget.Box | QWidget.Plain)

    def load_metrics(self):
        model = self.model_selector.currentText()
        if model:
            try:
                file_path = f"./models/{model}/val_metrics.csv"
                self.loaded_metrics = pd.read_csv(file_path, index_col=0)
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
                self.table_widget.setItem(row, col, QTableWidgetItem(str(value)))

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

    app = QApplication(sys.argv)
    visualizer = ModelMetricsVisualizer(models, 3)
    visualizer.show()
    sys.exit(app.exec_())