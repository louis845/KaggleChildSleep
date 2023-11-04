import sys
import os
from PySide2.QtWidgets import QApplication, QTableView, QFileDialog
from PySide2.QtCore import Qt, QAbstractTableModel
import pandas as pd

class PandasModel(QAbstractTableModel):
    def __init__(self, df = pd.DataFrame(), parent=None):
        QAbstractTableModel.__init__(self, parent=parent)
        self._df = df

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return None
        elif orientation == Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return None

    def data(self, index, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if not index.isValid():
            return None
        return str(self._df.iloc[index.row(), index.column()])

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=None):
        return len(self._df.index)

    def columnCount(self, parent=None):
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending= order == Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()

class DataFrameViewer(QTableView):
    def __init__(self, parent=None):
        super(DataFrameViewer, self).__init__(parent)
        self.setWindowTitle("DataFrame Viewer")
        self.setSortingEnabled(True)

    def show_df(self):
        filename = QFileDialog.getOpenFileName(self, "Open File", "../")[0]
        df = pd.read_parquet(filename)
        model = PandasModel(df)
        self.setModel(model)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dfviewer = DataFrameViewer()
    dfviewer.show_df()
    dfviewer.show()
    sys.exit(app.exec_())
