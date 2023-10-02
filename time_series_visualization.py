import sys
import os
import pandas as pd
from PySide2.QtWidgets import QApplication, QVBoxLayout, QFileDialog, QWidget, QHBoxLayout, QPushButton, QListWidget, QTabWidget, QListWidgetItem, QSplitter, QSlider, QLabel
from PySide2.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class MatplotlibWidget(QWidget):
    def __init__(self, df, title, events, parent=None):
        super(MatplotlibWidget, self).__init__(parent)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axis = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        # Slider and Label
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(10)
        self.slider.setMaximum(100)
        self.slider_label = QLabel("Window Size: 10")
        self.slider_label.setAlignment(Qt.AlignCenter)
        self.slider_label.setMaximumHeight(self.slider_label.sizeHint().height())
        self.layout.addWidget(self.slider_label)
        self.layout.addWidget(self.slider)
        self.slider.valueChanged.connect(self.update_plot)

        self.close_button = QPushButton("Close")
        self.layout.addWidget(self.close_button)

        self.df = df
        self.title = title
        self.events = events
        self.update_plot(self.slider.value())

    def update_plot(self, window):
        self.slider_label.setText(f"Window Size: {window}")
        self.plot_data(self.df, self.title, self.events, window)

    def plot_data(self, df, title, events, window):
        self.axis.clear()
        x = pd.to_datetime(df["timestamp"])  # Automatically parses the timestamp
        y1 = df["anglez"] / 35.52 # std computed by check_series_properties.py
        y2 = df["enmo"] / 0.1018 # std computed by check_series_properties.py
        self.axis.plot(x, y1, label="anglez")
        self.axis.plot(x, y2, label="enmo")

        # Plot running variance
        self.axis.plot(x, y1.rolling(window).var(), label="anglez variance")
        self.axis.plot(x, y2.rolling(window).var(), label="enmo variance")

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
        self.tab_widget = QTabWidget()

        self.splitter.addWidget(self.file_list)
        self.splitter.addWidget(self.tab_widget)

        # Set initial sizes of splitter
        self.splitter.setSizes([0.2 * self.width(), 0.8 * self.width()])

        self.layout.addWidget(self.splitter)

        self.file_list.itemDoubleClicked.connect(self.open_file)

        self.load_file_names()

    def load_file_names(self):
        for file in os.listdir("./individual_train_series"):
            if file.endswith(".parquet"):
                item = QListWidgetItem(file.replace(".parquet", ""))
                self.file_list.addItem(item)

    def open_file(self, item):
        filename = "./individual_train_series/" + item.text() + ".parquet"
        df = pd.read_parquet(filename)
        events = load_extra_events(item.text())
        plot_widget = MatplotlibWidget(df, item.text(), events)
        plot_widget.close_button.clicked.connect(lambda: self.close_tab(self.tab_widget.currentIndex()))
        self.tab_widget.addTab(plot_widget, item.text())

    def close_tab(self, index):
        self.tab_widget.removeTab(index)

def load_extra_events(series_id: str):
    events = pd.read_csv("data/train_events.csv")
    assert set(list(events["event"].unique())) == {"onset", "wakeup"}
    events = events.loc[events["series_id"] == series_id].dropna()
    assert len(events) > 0, series_id

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
