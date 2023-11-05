import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QCheckBox, QScrollArea, \
    QSplitter, QHBoxLayout
from PySide2.QtCore import Qt


class MetricPlotter(QMainWindow):
    def __init__(self, data_path):
        super(MetricPlotter, self).__init__()

        self.data_path = data_path
        self.models = []
        self.metrics = []
        self.model_checkboxes = []
        self.metric_checkboxes = []
        self.figures = []
        self.axes = []
        self.canvases = []

        self.initialize_ui()

    def initialize_ui(self):
        # Main widget
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        # Main layout
        main_layout = QVBoxLayout(main_widget)
        # Splitter for main and bottom area
        splitter_main_bottom = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter_main_bottom)

        # Main area for plots with a horizontal scroll bar
        self.plot_scroll = QScrollArea()
        self.plot_widget = QWidget()
        self.plot_layout = QHBoxLayout(self.plot_widget)
        self.plot_scroll.setWidget(self.plot_widget)
        self.plot_scroll.setWidgetResizable(True)
        self.plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        splitter_main_bottom.addWidget(self.plot_scroll)

        # Bottom area for model and metric selection
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        splitter_main_bottom.addWidget(bottom_widget)

        # Splitter for model and metric selection
        splitter_model_metric = QSplitter(Qt.Horizontal)
        bottom_layout.addWidget(splitter_model_metric)

        # Model selection area
        model_scroll = QScrollArea()
        self.model_widget = QWidget()
        self.model_layout = QVBoxLayout(self.model_widget)
        model_scroll.setWidget(self.model_widget)
        model_scroll.setWidgetResizable(True)
        splitter_model_metric.addWidget(model_scroll)

        # Metric selection area
        metric_scroll = QScrollArea()
        self.metric_widget = QWidget()
        self.metric_layout = QVBoxLayout(self.metric_widget)
        metric_scroll.setWidget(self.metric_widget)
        metric_scroll.setWidgetResizable(True)
        splitter_model_metric.addWidget(metric_scroll)

        # Plot button
        plot_button = QPushButton('Plot')
        plot_button.clicked.connect(self.plot_metrics)
        plot_button.setFixedHeight(plot_button.fontMetrics().height())
        bottom_layout.addWidget(plot_button)

        self.update_models()

    def update_models(self):
        models = sorted(os.listdir(self.data_path))
        for model in models:
            if os.path.isdir(os.path.join(self.data_path, model)):
                checkbox = QCheckBox(model)
                checkbox.stateChanged.connect(self.update_metrics)
                self.model_checkboxes.append(checkbox)
                self.model_layout.addWidget(checkbox)

    def update_metrics(self):
        previously_checked = []

        self.metrics.clear()
        for checkbox in self.metric_checkboxes:
            if checkbox.isChecked():
                previously_checked.append(checkbox.text())
            self.metric_layout.removeWidget(checkbox)
            checkbox.deleteLater()

        self.metric_checkboxes.clear()

        for model_checkbox in self.model_checkboxes:
            if model_checkbox.isChecked():
                model_metrics = self.get_metrics(model_checkbox.text())
                if not self.metrics:
                    self.metrics = model_metrics
                else:
                    self.metrics = list(set(self.metrics) & set(model_metrics))

        for metric in sorted(self.metrics):
            checkbox = QCheckBox(metric)
            self.metric_checkboxes.append(checkbox)
            self.metric_layout.addWidget(checkbox)

            if metric in previously_checked:
                checkbox.setChecked(True)

    def get_metrics(self, model):
        train_metrics = pd.read_csv(os.path.join(self.data_path, model, 'train_metrics.csv'),
                                    index_col=0).columns.tolist()
        val_metrics = pd.read_csv(os.path.join(self.data_path, model, 'val_metrics.csv'), index_col=0).columns.tolist()
        return train_metrics + val_metrics

    def plot_metrics(self):
        # Clear the previous plots
        for figure, ax, canvas in zip(self.figures, self.axes, self.canvases):
            figure.clf()
            self.plot_layout.removeWidget(canvas)
            del figure, ax, canvas
        self.figures.clear()
        self.axes.clear()
        self.canvases.clear()

        max_epochs = 0
        data = {}

        for model_checkbox in self.model_checkboxes:
            if model_checkbox.isChecked():
                model = model_checkbox.text()
                train_data = pd.read_csv(os.path.join(self.data_path, model, 'train_metrics.csv'), index_col=0)
                val_data = pd.read_csv(os.path.join(self.data_path, model, 'val_metrics.csv'), index_col=0)
                data[model] = pd.concat([train_data, val_data], axis=1)
                max_epochs = max(max_epochs, len(train_data), len(val_data))

        for metric_checkbox in self.metric_checkboxes:
            if metric_checkbox.isChecked():
                metric = metric_checkbox.text()

                # Create a new plot for each selected metric
                figure, ax = plt.subplots(1, 1, figsize=(5, 4))
                self.figures.append(figure)
                self.axes.append(ax)
                canvas = FigureCanvas(figure)
                canvas.setMinimumWidth(self.plot_scroll.width() / 2)
                self.canvases.append(canvas)
                self.plot_layout.addWidget(canvas)

                if "accuracy" in metric or "precision" in metric or "recall" in metric:
                    ax.set_ylim(0.75, 0.95)
                    ax.set_yticks(np.arange(0.75, 0.95, 0.05))

                for model, df in data.items():
                    epochs = np.arange(max_epochs)
                    values = df[metric].fillna(0)
                    values = np.pad(values, (0, max_epochs - len(values)))

                    ax.plot(epochs, values, label=model)

                ax.set_title(metric)
                ax.legend()

        self.plot_widget.adjustSize()


if __name__ == "__main__":
    app = QApplication([])
    plotter = MetricPlotter("models")
    plotter.show()
    app.exec_()