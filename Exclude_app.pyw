import os
import re
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from excludeloader import Ui_ExcludeLoader
from threading import Thread

find_all_finished = re.compile(r"All finished", re.S)

class MainWindow(QtWidgets.QMainWindow, Ui_ExcludeLoader):
    StatusChanged = QtCore.pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        main_window = self.setupUi(self)

        self.ex_order = 0
        self.case_num = 0
        self.per = 0
        self.per_ = 0
        self._per = 0
        self.case_fin = 0
        self.case_pre_run = []
        self.case_pre_exist = []

        from ProgressBar import PowerBar
        self.ProgressBar = PowerBar(80, self.centralwidget)
        self.ProgressBar.setGeometry(QtCore.QRect(40, 180, 511, 41))
        self.ProgressBar.set_value = 0
        self.Percentage.setText("")

        self.Load.clicked.connect(lambda event: self.load_signal())
        self.Confirm.clicked.connect(lambda event: self.preproceed_runner())
        self.timer = QtCore.QTimer()
        self.timer.setInterval(500)
        self.timer.stop()
        self.timer.timeout.connect(lambda *args: self.bar_calculator())

    def load_signal(self):
        try:
            self.FilePath.setText(FileBrowser().open_file_path_dialog())
        except IndexError:
            pass
        self.ProgressBar.set_value = 0
        self.Percentage.setText("")

    def preproceed_runner(self):

        os.makedirs(f"{self.FilePath.text()}\\banira_files", exist_ok=True)
        self.ex_order = 0
        self.ProgressBar.set_value = 0
        self.Percentage.setText("")
        self.Load.setEnabled(False)
        self.Confirm.setEnabled(False)
        self.FilePath.setEnabled(False)
        self.case_num = 0
        self.case_fin = 0
        self.per = 0
        self.per_ = 0
        self._per = 0

        self.case_pre_run = []
        self.case_pre_exist = []

        def exclude_file_runner():
            os.system(f"python .\\training_process\\preproceed_runner.pyw {self.FilePath.text()} {case}")

        for item in os.listdir(self.FilePath.text()):
            if os.path.isdir(self.FilePath.text() + '//' + item) and item != "banira_files":
                self.case_num += 1
                if not os.path.exists(f"{self.FilePath.text()}/banira_files/npy/{item}_xyz_predict.npy"):
                    self.case_pre_run.append(item)
        self.case_fin = self.case_num - len(self.case_pre_run)

        for item in os.listdir(f"{self.FilePath.text()}/banira_files/npy"):
            if item[-16:] == "_xyz_predict.npy":
                self.case_pre_exist.append(item[:-16])
        with open(f"{self.FilePath.text()}/banira_files/finished_ex.txt", "w") as f:
            for item in self.case_pre_exist:
                f.write(f"{item}\n")
            f.close()

        if not self.case_pre_run:
            self.ex_order = 1
            with open(f"{self.FilePath.text()}/banira_files/finished_ex.txt", "a") as f:
                f.write("All finished\n")

        else:
            case = """"""
            for item in self.case_pre_run:
                case += f"{item},"
            case = case[:-1]

        if self.ex_order == 1:
            self.ProgressBar.set_value = 100
            self.Percentage.setText("100%")
            self.Load.setEnabled(True)
            self.Confirm.setEnabled(True)
            self.FilePath.setEnabled(True)
        else:
            ex_run = Thread(target=exclude_file_runner)
            ex_run.setDaemon(True)
            ex_run.start()
            self.timer.start()

    def bar_calculator(self):
        self.per_ += 0.025
        buf_ex = []
        try:
            with open(f"{self.FilePath.text()}\\banira_files\\finished_ex.txt", "r") as f:
                for row in f:
                    buf_ex.append(row)
                f.close()
            if len(buf_ex) < len(self.case_pre_exist) + len(self.case_pre_run) + 1:
                self.per = round((len(buf_ex) - len(self.case_pre_exist) + self.case_fin) / self.case_num * 100)
                if self._per < self.per:
                    self._per = self.per
                    self.per_ = self.per
                self.ProgressBar.set_value = round(self.per_)
                self.Percentage.setText(f"{round(self.per_)}%")
            else:
                self.per_ = 100
        except FileNotFoundError:
            pass
        self.ProgressBar.set_value = round(self.per_)
        self.Percentage.setText(f"{round(self.per_)}%")
        try:
            assert round(self.per_) < 100
        except AssertionError:
            self.ProgressBar.set_value = 100
            self.Percentage.setText("100%")
            self.timer.stop()
            self.Load.setEnabled(True)
            self.Confirm.setEnabled(True)
            self.FilePath.setEnabled(True)

class TabBar(QtWidgets.QTabBar):

    def tabSizeHint(self, index: int) -> QtCore.QSize:
        s = QtWidgets.QTabBar.tabSizeHint(self, index)
        # s.transpose()
        return s

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        painter = QtWidgets.QStylePainter(self)
        style_option = QtWidgets.QStyleOptionTab()

        for i in range(self.count()):
            self.initStyleOption(style_option, i)
            painter.drawControl(QtWidgets.QStyle.CE_TabBarTabShape, style_option)
            painter.save()

            size = style_option.rect.size()
            size.transpose()
            rect = QtCore.QRect(QtCore.QPoint(), size)
            rect.moveCenter(style_option.rect.center())
            style_option.rect = rect

            center = self.tabRect(i).center()
            painter.translate(center)
            painter.rotate(90)
            painter.translate(-center)
            painter.drawControl(QtWidgets.QStyle.CE_TabBarTabLabel, style_option)
            painter.restore()


class MyTabWidget(QtWidgets.QTabWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setTabBar(TabBar(self))
        self.setTabPosition(QtWidgets.QTabWidget.West)

class FileBrowser(QtWidgets.QWidget):
    def __init__(self):
        super(FileBrowser, self).__init__()

    def open_file_path_dialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        options |= QtWidgets.QFileDialog.ShowDirsOnly
        path = QtWidgets.QFileDialog.getExistingDirectory(self, options=options)
        return path

    def open_file_name_dialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name = QtWidgets.QFileDialog.getOpenFileNames(self, options=options)
        return file_name


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()