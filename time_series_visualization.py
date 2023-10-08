import sys
import os
import pandas as pd
import numpy as np
from PySide2.QtWidgets import QApplication, QVBoxLayout, QFileDialog, QWidget, QHBoxLayout, QPushButton, QListWidget, QTabWidget, QListWidgetItem, QSplitter, QSlider, QLabel, QCheckBox, QComboBox
from PySide2.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class SliderWithDisplayWidget(QWidget):
    def __init__(self, slider_name, slider_min, slider_max):
        super(SliderWithDisplayWidget, self).__init__()

        self.layout = QVBoxLayout(self)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(slider_min)
        self.slider.setMaximum(slider_max)
        self.slider.setValue(slider_min)
        self.slider_label = QLabel(slider_name + ": " + str(slider_min))
        self.slider_label.setAlignment(Qt.AlignCenter)
        self.slider_label.setMaximumHeight(self.slider_label.sizeHint().height())
        self.layout.addWidget(self.slider_label)
        self.layout.addWidget(self.slider)

        self.slider.valueChanged.connect(self.update_label)
        self.slider_name = slider_name

    def get_value(self):
        return self.slider.value()

    def update_label(self, value):
        self.slider_label.setText(self.slider_name + ": " + str(value))

class MatplotlibWidget(QWidget):
    def __init__(self, df, title, events, pred_probas, parent=None):
        super(MatplotlibWidget, self).__init__(parent)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axis = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        # Slider and Label
        self.variance_slider = SliderWithDisplayWidget("Variance window", 100, 2000)
        self.variance_median_slider = SliderWithDisplayWidget("Variance median window", 100, 2000)
        self.layout.addWidget(self.variance_slider)
        self.layout.addWidget(self.variance_median_slider)

        self.show_data_checkbox = QCheckBox("Show Data")
        self.layout.addWidget(self.show_data_checkbox)
        self.show_data_checkbox.setChecked(True)

        self.show_variance_checkbox = QCheckBox("Show Variance")
        self.layout.addWidget(self.show_variance_checkbox)
        self.show_variance_checkbox.setChecked(True)

        self.show_variance_median_checkbox = QCheckBox("Show Variance Median")
        self.layout.addWidget(self.show_variance_median_checkbox)
        self.show_variance_median_checkbox.setChecked(False)

        self.update_button = QPushButton("Update")
        self.layout.addWidget(self.update_button)
        self.update_button.clicked.connect(lambda: self.plot_data(self.df, self.title, self.events))

        self.close_button = QPushButton("Close")
        self.layout.addWidget(self.close_button)

        self.df = df
        self.title = title
        self.events = events
        self.pred_probas = pred_probas
        self.plot_data(self.df, self.title, self.events)

    def plot_data(self, df, title, events):
        window = self.variance_slider.get_value()
        window2 = self.variance_median_slider.get_value()
        show_data = self.show_data_checkbox.isChecked()
        show_variance = self.show_variance_checkbox.isChecked()
        show_variance_median = self.show_variance_median_checkbox.isChecked()

        self.axis.clear()
        x = pd.to_datetime(df["timestamp"])  # Automatically parses the timestamp
        y1 = df["anglez"] / 35.52 # std computed by check_series_properties.py
        y2 = df["enmo"] / 0.1018 # std computed by check_series_properties.py
        if show_data:
            self.axis.plot(x, y1, label="anglez")
            self.axis.plot(x, y2, label="enmo")

        # Plot running variance
        y1_var = y1.rolling(window).var()
        y2_var = y2.rolling(window).var()
        if show_variance:
            self.axis.plot(x, y1_var, label="anglez variance")
            self.axis.plot(x, y2_var, label="enmo variance")

        # Plot running variance median
        y1_var_median = y1_var.rolling(window2).median()
        y2_var_median = y2_var.rolling(window2).median()
        if show_variance_median:
            self.axis.plot(x, y1_var_median, label="anglez variance median")
            self.axis.plot(x, y2_var_median, label="enmo variance median")

        if self.pred_probas is not None:
            self.axis.plot(x, self.pred_probas, label="Probas")

        for event_time, event_type in events:
            color = "blue" if event_type == 1 else "red"
            self.axis.axvline(pd.to_datetime(event_time), color=color, alpha=0.5, linestyle="--")

        self.axis.set_title(title)
        self.axis.legend()
        self.canvas.draw()

class MainWidget(QWidget):
    def __init__(self, parent=None):
        super(MainWidget, self).__init__(parent)

        self.setWindowTitle("Visualization")
        self.resize(1280, 720)

        self.layout = QHBoxLayout(self)
        self.splitter = QSplitter(Qt.Horizontal)

        self.file_list = QListWidget()
        self.main_layout = QVBoxLayout()

        self.splitter.addWidget(self.file_list)
        self.splitter.addWidget(self.main_layout)

        # Set initial sizes of splitter
        self.splitter.setSizes([0.2 * self.width(), 0.8 * self.width()])

        self.layout.addWidget(self.splitter)

        self.file_list.itemDoubleClicked.connect(self.open_file)

        self.load_file_names()

        # Tab widget and dropdown
        self.dropdown_list = QComboBox()
        self.main_layout.addWidget(self.dropdown_list)
        self.dropdown_list.addItem("None")
        if os.path.isdir("pseudo_labels"):
            for folder in os.listdir("pseudo_labels"):
                self.dropdown_list.addItem(folder)
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)

    def load_file_names(self):
        file_items = []
        for file in os.listdir("./individual_train_series"):
            if file.endswith(".parquet"):
                file_items.append(file.replace(".parquet", ""))
        file_items.sort()
        for file in file_items:
            item = QListWidgetItem(file)
            self.file_list.addItem(item)

    def open_file(self, item):
        filename = "./individual_train_series/" + item.text() + ".parquet"
        df = pd.read_parquet(filename)
        events = load_extra_events(item.text())

        # load pred probas if selected
        pred_probas = None
        if self.dropdown_list.currentText() != "None":
            pred_probas = self.dropdown_list.currentText()
            if os.path.isfile(os.path.join("pseudo_labels", pred_probas, item.text() + ".npy")):
                pred_probas = np.load(os.path.join("pseudo_labels", pred_probas, item.text() + ".npy"))

        plot_widget = MatplotlibWidget(df, item.text(), events, pred_probas)
        plot_widget.close_button.clicked.connect(lambda: self.close_tab(self.tab_widget.currentIndex()))
        self.tab_widget.addTab(plot_widget, item.text())

    def close_tab(self, index):
        self.tab_widget.removeTab(index)

def load_extra_events(series_id: str):
    events = pd.read_csv("data/train_events.csv")
    assert set(list(events["event"].unique())) == {"onset", "wakeup"}
    events = events.loc[events["series_id"] == series_id].dropna()
    if len(events) == 0:
        return []

    events_list = []
    for k in range(len(events)):
        event = events.iloc[k]
        type = 1 if (event["event"] == "onset") else 2
        events_list.append((event["timestamp"], type))
    return events_list

if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_widget = MainWidget()
    main_widget.show()
    sys.exit(app.exec_())
