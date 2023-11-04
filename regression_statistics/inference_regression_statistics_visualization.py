import sys
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QCheckBox, QPushButton, QGroupBox, QHBoxLayout, QLabel, QFrame, QSplitter
from PySide2.QtCore import Qt

class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(5, 4), dpi=100)
        self.axes = fig.add_subplot(111)
        super(MyMplCanvas, self).__init__(fig)

class ApplicationWindow(QWidget):
    def __init__(self, data_statistics, A, B):
        super().__init__()
        self.data_statistics = data_statistics
        self.A = A
        self.B = B

        self.main_layout = QVBoxLayout()

        self.init_ui()

    def init_ui(self):
        self.kde_plot = MyMplCanvas(self)
        self.hist_plot = MyMplCanvas(self)
        self.percentile_plot = MyMplCanvas(self)

        plots_widget = QWidget(self)
        plots_layout = QHBoxLayout()
        plots_widget.setLayout(plots_layout)

        # kde plot
        kde_widget = QFrame(plots_widget)
        kde_layout = QVBoxLayout()
        kde_widget.setLayout(kde_layout)

        kde_label = QLabel("KDE")
        kde_label.setAlignment(Qt.AlignCenter)
        kde_label.setFixedHeight(kde_label.fontMetrics().boundingRect(kde_label.text()).height())
        kde_toolbar = NavigationToolbar2QT(self.kde_plot, kde_widget)
        kde_layout.addWidget(kde_label)
        kde_layout.addWidget(self.kde_plot)
        kde_layout.addWidget(kde_toolbar)
        kde_widget.setFrameShape(QFrame.StyledPanel)

        # hist plot
        hist_widget = QFrame(plots_widget)
        hist_layout = QVBoxLayout()
        hist_widget.setLayout(hist_layout)

        hist_label = QLabel("Histogram")
        hist_label.setAlignment(Qt.AlignCenter)
        hist_label.setFixedHeight(hist_label.fontMetrics().boundingRect(hist_label.text()).height())
        hist_toolbar = NavigationToolbar2QT(self.hist_plot, hist_widget)
        hist_layout.addWidget(hist_label)
        hist_layout.addWidget(self.hist_plot)
        hist_layout.addWidget(hist_toolbar)
        hist_widget.setFrameShape(QFrame.StyledPanel)

        # percentile plot
        percentile_widget = QFrame(plots_widget)
        percentile_layout = QVBoxLayout()
        percentile_widget.setLayout(percentile_layout)

        percentile_label = QLabel("Percentile")
        percentile_label.setAlignment(Qt.AlignCenter)
        percentile_label.setFixedHeight(percentile_label.fontMetrics().boundingRect(percentile_label.text()).height())
        percentile_toolbar = NavigationToolbar2QT(self.percentile_plot, percentile_widget)
        percentile_layout.addWidget(percentile_label)
        percentile_layout.addWidget(self.percentile_plot)
        percentile_layout.addWidget(percentile_toolbar)
        percentile_widget.setFrameShape(QFrame.StyledPanel)

        # add plots to plots_layout
        plots_layout.addWidget(kde_widget)
        plots_layout.addWidget(hist_widget)
        plots_layout.addWidget(percentile_widget)

        self.top_bottom_splitter = QSplitter(Qt.Vertical)
        self.top_bottom_splitter.addWidget(plots_widget)

        self.main_layout.addWidget(self.top_bottom_splitter)

        self.init_checkboxes()
        self.init_plot_button()

        self.setLayout(self.main_layout)

    def init_checkboxes(self):
        self.checkboxes_A = {a: QCheckBox(str(a), self) for a in self.A}
        self.checkboxes_B = {b: QCheckBox(str(b), self) for b in self.B}

        self.checkboxes_widget = QWidget(self)
        self.checkbox_layout = QHBoxLayout()
        self.checkboxes_widget.setLayout(self.checkbox_layout)


        group_A = QGroupBox("Set A")
        layout_A = QVBoxLayout()
        for checkbox in self.checkboxes_A.values():
            layout_A.addWidget(checkbox)
        group_A.setLayout(layout_A)

        group_B = QGroupBox("Set B")
        layout_B = QVBoxLayout()
        for checkbox in self.checkboxes_B.values():
            layout_B.addWidget(checkbox)
        group_B.setLayout(layout_B)

        self.checkbox_layout.addWidget(group_A)
        self.checkbox_layout.addWidget(group_B)

        self.top_bottom_splitter.addWidget(self.checkboxes_widget)

    def init_plot_button(self):
        plot_button = QPushButton("Plot", self)
        plot_button.clicked.connect(self.plot_data)
        self.main_layout.addWidget(plot_button)

    def plot_data(self):
        self.kde_plot.axes.clear()
        self.hist_plot.axes.clear()
        self.percentile_plot.axes.clear()

        for a, checkbox_A in self.checkboxes_A.items():
            for b, checkbox_B in self.checkboxes_B.items():
                if checkbox_A.isChecked() and checkbox_B.isChecked():
                    data = self.data_statistics[a][b]
                    label = str(a) + " " + str(b)
                    sns.kdeplot(data, ax=self.kde_plot.axes, label=label)
                    sns.histplot(data, ax=self.hist_plot.axes, label=label)
                    percentiles = np.percentile(data, np.linspace(0, 100, num=101))
                    self.percentile_plot.axes.plot(np.linspace(0, 100, num=101), percentiles, label=label)

        for percentile in [25, 50, 75]:
            self.percentile_plot.axes.axvline(x=percentile, color='r', linestyle='--')

        self.kde_plot.axes.legend()
        self.hist_plot.axes.legend()
        self.percentile_plot.axes.legend()
        self.kde_plot.draw()
        self.hist_plot.draw()
        self.percentile_plot.draw()
