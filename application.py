import time
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
import numpy as np
from PyQt5 import QtWidgets, uic, QtCore, QtGui
from PyQt5.QtWidgets import QOpenGLWidget
import pyqtgraph as pg
from mainwindow import Ui_MainWindow
from warningwindow import Ui_WarningWindow
import sys  # We need sys so that we can pass argv to QApplication
import os
from PIL import ImageQt, Image
from threading import Thread
import re
from datetime import timedelta
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as mplot

find_all_finished = re.compile(r"All finished", re.S)
find_finished_training = re.compile(r"Training Finished", re.S)
find_stopped_training = re.compile(r"Training Interrupted", re.S)


class Status:
    def __init__(self):
        # Model Selection
        self.model = None

        # Data Fitting
        self.data_address = None

        # Training Process
        self.training = None

        # Test and Visualization
        self.case_address = None
        self.projection = None
        self.translation = None
        self.view = None
        self.cursor = None

        # Implementation


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    StatusChanged = QtCore.pyqtSignal(bool)

    def __init__(self, clip_board, status=Status(), *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.total_quantite = None
        main_window = self.setupUi(self)
        self.status = status
        # self.setWindowIcon(QtGui.QIcon(r"ばにら.png"))

        self.clip_board = clip_board

        os.makedirs("./output_csv", exist_ok=True)
        # signal parameters for training tab
        self.ex_order = 0
        self.tr_order = 0
        self.case_int = 0
        self.model_int = 0
        self.case_path = []
        self.model_path = []
        self.test_int = 0
        self.model_signal = 0
        self.test_case = "sts"
        self.case_num = 0
        self.per = 0
        self.per_ = 0
        self._per = 0
        self.PS_signal = 0
        self.train_int = 0
        self.tr_case = """"""

        # global initial settings
        self.tabWidget.setCurrentIndex(0)

        self.tabWidget.setTabEnabled(0, True)
        self.tabWidget.setTabEnabled(1, False)
        self.tabWidget.setTabEnabled(2, False)
        self.tabWidget.setTabEnabled(3, False)
        self.tabWidget.setTabEnabled(4, False)

        # button change
        self.Pause.clicked.connect(
            lambda event: self.Pause.setText("Continue") if event else self.Pause.setText("Pause")
        )
        self.Lock.clicked.connect(
            lambda event: self.Lock.setText("Unlock") if event else self.Lock.setText("Lock")
        )
        self.OutputConfirm.clicked.connect(
            lambda event: self.OutputConfirm.setText("Cancel") if event else self.OutputConfirm.setText("Confirm")
        )
        self.OutputConfirm2.clicked.connect(
            lambda event: self.OutputConfirm2.setText("Cancel") if event else self.OutputConfirm2.setText("Confirm")
        )

        self.Confirm2.clicked.connect(
            lambda event: self.Confirm2.setText("Cancel") if event else self.Confirm2.setText("Confirm")
        )
        self.Confirm3.clicked.connect(
            lambda event: self.Confirm3.setText("Cancel") if event else self.Confirm3.setText("Confirm")
        )

        # progress bar initial settings
        from ProgressBar import PowerBar
        self.ProgressBar = PowerBar(80, self.DataFit)
        self.ProgressBar.setGeometry(QtCore.QRect(30, 130, 751, 41))
        self.ProgressBar.set_value = 0
        self.Percentage.setText("")
        self.TrainingLogBox.setText("")

        # font center settings
        self.FilePath.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.EpochValue.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # training graph initial settings
        self.DataLogBox.setText("")
        labelStyle = {'color': '#4e526b', 'font': '12pt Dubai'}
        self.customized_graph = pg.PlotWidget(self.GraphWidget)
        self.customized_graph.setGeometry(QtCore.QRect(0, 0, 781, 401))
        self.customized_graph.setBackground((212, 218, 236))
        self.customized_graph.getAxis('left').setPen(pg.mkPen((78, 81, 107)))
        self.customized_graph.getAxis('left').setTextPen(pg.mkPen((78, 81, 107)))
        self.customized_graph.getAxis('left').setTickFont(QtGui.QFont('Dubai', 10))
        self.customized_graph.setLabel('left', "Loss", **labelStyle)

        self.customized_graph.getAxis('bottom').setPen(pg.mkPen((78, 81, 107)))
        self.customized_graph.getAxis('bottom').setTextPen(pg.mkPen((78, 81, 107)))
        self.customized_graph.getAxis('bottom').setTickFont(QtGui.QFont('Dubai', 10))
        self.customized_graph.setLabel('bottom', "Epoch", **labelStyle)
        self.legend = pg.LegendItem((80, 60), offset=(620, 5))
        self.legend.setParentItem(self.customized_graph.graphicsItem())
        self.label_signal = 0
        self.customized_graph.hide()


        # model selection tab initial settings
        self.Model1Opt1.clicked.connect(lambda event: self.m2_opt1_clb())
        self.Model1Opt2.clicked.connect(lambda event: self.m2_opt23_clb(1))
        self.Model1Opt3.clicked.connect(lambda event: self.m2_opt23_clb(2))
        self.Model1.setChecked(True)
        self.Model2Opt1.setEnabled(False)
        self.Model2Opt2.setEnabled(False)
        self.Model2Opt3.setEnabled(False)
        self.Model1.clicked.connect(self.button_clicked_clb)
        self.Model2.clicked.connect(self.button_clicked_clb2)
        self.Lock.clicked.connect(self.lock_buttons_clb)

        # data fitting tab initial settings
        self.TrainCase.hide()
        self.TrainCaseText.hide()
        self.bgwidget4.hide()
        self.Warning3.setText("")
        self.bgwidget3.hide()
        self.TestCase.hide()
        self.TestCaseText.hide()
        self.Confirm2.hide()
        self.TrainRatio.textEdited.connect(lambda event: self.ratio_calculator())
        self.Load.clicked.connect(lambda event: self.load_signal())
        self.Confirm.clicked.connect(lambda event: self.interpolation_trigger())
        self.timer = QtCore.QTimer()
        self.timer.setInterval(200)
        self.timer.stop()
        self.timer.timeout.connect(lambda *args: self.data_log_printer())
        self.Confirm2.clicked.connect(lambda event: self.test_confirm_clb())
        self.Confirm3.clicked.connect(lambda event: self.test_confirm_clb_2())
        self.TrainCase.clicked.connect(lambda event: self.train_test_connection())

        # training tab initial settings
        self.TrainCase.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.Start.setEnabled(False)
        self.Pause.setEnabled(False)
        self.Stop.setEnabled(False)
        self.ProgressNoticer.setText("")
        self.ImpleNoticer.setText("")
        self.Start.clicked.connect(lambda event: self.discrate_classifier_trigger())
        self.timer2 = QtCore.QTimer()
        self.timer2.setInterval(200)
        self.timer2.stop()
        self.timer2.timeout.connect(lambda *args: self.training_log_printer())
        self.GraphWidget = pg.PlotWidget()

        self.timer3 = QtCore.QTimer()
        self.timer3.setInterval(1000)
        self.timer3.stop()
        self.timer3.timeout.connect(lambda *args: self.training_graph_printer())

        self.timer4 = QtCore.QTimer()
        self.timer4.setInterval(500)
        self.timer4.stop()
        self.timer4.timeout.connect(lambda *args: self.training_finish_order())

        self.Pause.clicked.connect(lambda event: self.training_pause())
        self.Stop.clicked.connect(lambda event: self.training_stop())

        # implementation tab initial settings
        self.Warning.setText("")
        self.Warning2.setText("")
        self.OutputLoad.clicked.connect(lambda event: self.load_signal_case())
        self.OutputLoad3.clicked.connect(lambda event: self.load_signal_case_2())
        self.OutputLoad2.clicked.connect(lambda event: self.load_signal_model())
        self.OutputLoad4.clicked.connect(lambda event: self.load_signal_model_2())
        self.OutputConfirm.clicked.connect(self.case_confirm_clb)
        self.OutputConfirm2.clicked.connect(self.model_confirm_clb)
        self.Compute.clicked.connect(lambda *args: self.predict_trigger())
        self.Finished.setText("")
        self.timer5 = QtCore.QTimer()
        self.timer5.setInterval(500)
        self.timer5.stop()
        self.timer5.timeout.connect(lambda *args: self.predict_finish_order())

        # visualization tab initial settings
        self.ViewPortStatus = {"Projection": 0,
                               "View": 0}
        self.Xview.clicked.connect(lambda event: self.x_lock_clb())
        self.Yview.clicked.connect(lambda event: self.y_lock_clb())
        self.Zview.clicked.connect(lambda event: self.z_lock_clb())

        self.ViewPort.setMinimumSize(621, 591)
        self.glw = MyGLWidget(self.ViewPort, self)
        self.glw.resize(621, 591)
        self.glw.makeCurrent()
        self.XYZ.clicked.connect(lambda event: self.axis_clb())
        self.ViewPortLog.setText("")
        self.ViewIndex.setText("{:.2f},{:.2f},{:.2f}\n{:.2f},{:.2f},{:.2f}\n{:.2f},{:.2f},{:.2f}".format(*self.glw.camera.position.xyz, *self.glw.camera.front.xyz, *self.glw.camera.up.xyz))
        self.Set.clicked.connect(lambda event: self.camera_set())
        self.Copy.clicked.connect(lambda event: self.copy_clb())
        self.Save.clicked.connect(lambda event: self.save_clb())

    def ViewIndex_setText_clb(self):
        self.ViewIndex.setText(
            "{:.2f},{:.2f},{:.2f}\n{:.2f},{:.2f},{:.2f}\n{:.2f},{:.2f},{:.2f}".format(*self.glw.camera.position.xyz, *self.glw.camera.front.xyz, *self.glw.camera.up.xyz))

    def m2_opt1_clb(self):
        self.TrainCase.hide()
        self.TrainCaseText.hide()
        self.bgwidget4.hide()

        self.RatioLabel.show()
        self.RatioLabel2.show()
        self.RatioLabel3.show()
        self.RatioLabel4.show()
        self.TrainRatio.show()
        self.TestRatio.show()
        self.Confirm3.show()

        self.bgwidget3.hide()
        self.TestCase.hide()
        self.TestCase.clear()
        self.TrainCase.clear()
        self.TestCaseText.hide()
        self.TestCase.setEnabled(True)
        self.TrainCase.setEnabled(True)
        self.Confirm2.hide()
        self.Warning3.setText("")
        self.ProgressBar.set_value = 0
        self.Percentage.setText("")
        self.DataLogBox.setText("")
        self.tabWidget.setTabEnabled(2, False)
        self.model_signal = 0
        self.Confirm2.setChecked(False)
        self.Confirm3.setChecked(False)
        self.Load.setEnabled(True)
        self.Confirm.setEnabled(True)

    def m2_opt23_clb(self, number):
        self.TrainCase.hide()
        self.TrainCaseText.hide()
        self.bgwidget4.hide()

        self.RatioLabel.hide()
        self.RatioLabel2.hide()
        self.RatioLabel3.hide()
        self.RatioLabel4.hide()
        self.TrainRatio.hide()
        self.TestRatio.hide()
        self.Confirm3.hide()

        self.bgwidget3.show()
        self.TestCase.show()
        self.TestCase.clear()
        self.TestCase.setEnabled(True)
        self.TrainCase.setEnabled(True)
        self.TrainCase.clear()
        self.TestCaseText.show()
        self.Confirm2.show()
        self.Warning3.setText("")
        self.Confirm2.setChecked(False)
        self.ProgressBar.set_value = 0
        self.Percentage.setText("")
        self.DataLogBox.setText("")
        self.tabWidget.setTabEnabled(2, False)
        self.model_signal = number
        self.Confirm2.setChecked(False)
        self.Confirm3.setChecked(False)
        self.Load.setEnabled(True)
        self.Confirm.setEnabled(True)

    def ratio_calculator(self):
        try:
            int(self.TrainRatio.text())
            if 0 < int(self.TrainRatio.text()) < 100:
                self.TestRatio.setText(str(100 - int(self.TrainRatio.text())))
            elif int(self.TrainRatio.text()) >= 100:
                self.TrainRatio.setText("99")
                self.TestRatio.setText("1")
            elif int(self.TrainRatio.text()) <= 0:
                self.TrainRatio.setText("1")
                self.TestRatio.setText("99")
        except ValueError:
            self.TrainRatio.setText("")
            self.TestRatio.setText("")

    def lock_buttons_clb(self, event):
        self.Model1.setChecked(self.Model1.isChecked())
        self.Model1.setEnabled(not event)
        self.Model2.setChecked(self.Model2.isChecked())
        self.Model2.setEnabled(not event)
        self.Model1Opt1.setChecked(self.Model1Opt1.isChecked())
        self.Model1Opt1.setEnabled(not event and self.Model1.isChecked())
        self.Model1Opt2.setChecked(self.Model1Opt2.isChecked())
        self.Model1Opt2.setEnabled(not event and self.Model1.isChecked())
        self.Model1Opt3.setChecked(self.Model1Opt3.isChecked())
        self.Model1Opt3.setEnabled(not event and self.Model1.isChecked())
        self.Model2Opt1.setChecked(self.Model2Opt1.isChecked())
        self.Model2Opt1.setEnabled(not event and self.Model2.isChecked())
        self.Model2Opt2.setChecked(self.Model2Opt2.isChecked())
        self.Model2Opt2.setEnabled(not event and self.Model2.isChecked())
        self.Model2Opt3.setChecked(self.Model2Opt3.isChecked())
        self.Model2Opt3.setEnabled(not event and self.Model2.isChecked())
        self.tabWidget.setTabEnabled(1, event)
        self.tabWidget.setTabEnabled(3, event)

    # button status functions
    def button_clicked_clb(self, event):
        if event and self.status.model == "AlexNet":
            pass
        elif event:
            self.status.model = "AlexNet"
            self.Model1Opt1.setChecked(self.Model1Opt1.isChecked())
            self.Model1Opt1.setEnabled(event)
            self.Model1Opt2.setChecked(self.Model1Opt2.isChecked())
            self.Model1Opt2.setEnabled(event)
            self.Model1Opt3.setChecked(self.Model1Opt3.isChecked())
            self.Model1Opt3.setEnabled(event)
            self.Model2Opt1.setChecked(self.Model2Opt1.isChecked())
            self.Model2Opt1.setEnabled(not event)
            self.Model2Opt2.setChecked(self.Model2Opt2.isChecked())
            self.Model2Opt2.setEnabled(not event)
            self.Model2Opt3.setChecked(self.Model2Opt3.isChecked())
            self.Model2Opt3.setEnabled(not event)

    def button_clicked_clb2(self, event):
        if event and self.status.model == "Model2":
            pass
        elif event:
            self.status.model = "Model2"
            self.Model1Opt1.setChecked(self.Model1Opt1.isChecked())
            self.Model1Opt1.setEnabled(not event)
            self.Model1Opt2.setChecked(self.Model1Opt2.isChecked())
            self.Model1Opt2.setEnabled(not event)
            self.Model1Opt3.setChecked(self.Model1Opt3.isChecked())
            self.Model1Opt3.setEnabled(not event)
            self.Model2Opt1.setChecked(self.Model2Opt1.isChecked())
            self.Model2Opt1.setEnabled(event)
            self.Model2Opt2.setChecked(self.Model2Opt2.isChecked())
            self.Model2Opt2.setEnabled(event)
            self.Model2Opt3.setChecked(self.Model2Opt3.isChecked())
            self.Model2Opt3.setEnabled(event)

    def load_signal(self):
        try:
            self.FilePath.setText(FileBrowser().open_file_path_dialog())
        except IndexError:
            pass
        self.TestCase.clear()
        self.tabWidget.setTabEnabled(2, False)
        self.tabWidget.setTabEnabled(4, False)
        self.DataLogBox.setText("")
        self.ProgressBar.set_value = 0
        self.Percentage.setText("")
        self.Confirm2.setChecked(False)
        self.Confirm3.setChecked(False)

    def test_confirm_clb(self):
        if self.TestCase.currentRow() == -1:
            self.Warning3.setText("*Please choose a test case.")
            self.Confirm2.setChecked(False)
            self.Confirm2.setText("Confirm")
        else:
            self.tr_case = """"""
            self.Warning3.setText("")
            for i in range(len(self.TrainCase.selectedItems())):
                self.tr_case += f"{self.TrainCase.selectedItems()[i].text()},"
            self.test_case = self.TestCase.currentItem().text()
        self.tabWidget.setTabEnabled(2, self.Confirm2.isChecked())
        self.TrainCase.setEnabled(not self.Confirm2.isChecked())
        self.TestCase.setEnabled(not self.Confirm2.isChecked())
        self.Load.setEnabled(not self.Confirm2.isChecked())
        self.Confirm.setEnabled(not self.Confirm2.isChecked())

    def test_confirm_clb_2(self):
        self.tr_case = """"""
        for i in range(len(self.TrainCase.selectedItems())):
            self.tr_case += f"{self.TrainCase.selectedItems()[i].text()},"
        self.TrainRatio.setReadOnly(self.Confirm3.isChecked())
        self.tabWidget.setTabEnabled(2, self.Confirm3.isChecked())
        self.TrainCase.setEnabled(not self.Confirm3.isChecked())
        self.TestCase.setEnabled(not self.Confirm3.isChecked())
        self.Load.setEnabled(not self.Confirm3.isChecked())
        self.Confirm.setEnabled(not self.Confirm3.isChecked())

    def train_test_connection(self):
        self.TestCase.clear()
        self.test_int = 0
        if self.Model1Opt2.isChecked():
            for i in range(self.train_int):
                if not self.TrainCase.item(i).isSelected():
                    self.TestCase.insertItem(self.test_int, self.TrainCase.item(i).text())
                    self.test_int += 1
        elif self.Model1Opt3.isChecked():
            for i in range(self.train_int):
                if self.TrainCase.item(i).isSelected():
                    self.TestCase.insertItem(self.test_int, self.TrainCase.item(i).text())
                    self.test_int += 1

    def training_pause(self):
        if self.Pause.isChecked():
            open(f"{self.FilePath.text()}/banira_files\\pause", "w").write("0")
        else:
            os.remove(f"{self.FilePath.text()}/banira_files\\pause")

    def training_stop(self):
        if self.Pause.isChecked():
            os.remove(f"{self.FilePath.text()}/banira_files\\pause")
            self.Pause.setChecked(False)
            self.Pause.setText("Pause")
        open(f"{self.FilePath.text()}/banira_files\\stop", "w").write("0")
        self.Pause.setEnabled(False)
        self.Stop.setEnabled(False)


    def load_signal_case(self):
        try:
            path, filename = os.path.split(FileBrowser().open_file_name_dialog()[0][0])
            if not filename[-16:] == "_xyz_predict.npy":
                dlg = MyDialog2(self, "Please load _predict.npy file.")
                dlg.setWindowTitle("Warning")
                dlg.exec()
            else:
                self.CaseOption.insertItem(self.case_int, filename[:-16])
                self.case_path.append(path + "\\" + filename)
                self.case_int += 1
        except IndexError:
            pass

    def load_signal_case_2(self):
        try:
            filepath = FileBrowser().open_file_path_dialog()
            for item in os.listdir(filepath):
                if item[-16:] == "_xyz_predict.npy":
                    self.CaseOption.insertItem(self.case_int, item[:-16])
                    self.case_path.append(filepath + "\\" + item)
                    self.case_int += 1
        except IndexError:
            pass

    def load_signal_model(self):
        try:
            path, filename = os.path.split(FileBrowser().open_file_name_dialog()[0][0])
            if not re.findall(re.compile(r"to", re.S), filename[:7]) and not filename[:7] == "TumourC":
                dlg = MyDialog2(self, "Please load TumourClassifier file.")
                dlg.setWindowTitle("Warning")
                dlg.exec()
            else:
                self.ModelOption.insertItem(self.model_int, filename)
                self.model_path.append(path + "\\" + filename)
                self.model_int += 1
        except IndexError:
            pass

    def load_signal_model_2(self):
        try:
            filepath = FileBrowser().open_file_path_dialog()
            for item in os.listdir(filepath):
                if re.findall(re.compile(r"to", re.S), item[:7]) or item[:7] == "TumourC":
                    self.ModelOption.insertItem(self.model_int, item)
                    self.model_path.append(filepath + "\\" + item)
                    self.model_int += 1
        except IndexError:
            pass

    def case_confirm_clb(self, event):
        if self.CaseOption.currentRow() == -1:
            self.Warning.setText("*Please choose a case.")
            self.OutputConfirm.setChecked(False)
            self.OutputConfirm.setText("Confirm")
        else:
            self.Warning.setText("")
        self.CaseOption.setEnabled(not self.OutputConfirm.isChecked())
        self.OutputLoad.setEnabled(not self.OutputConfirm.isChecked())
        self.OutputLoad3.setEnabled(not self.OutputConfirm.isChecked())


    def model_confirm_clb(self, event):
        if self.ModelOption.currentRow() == -1:
            self.Warning2.setText("*Please choose a model.")
            self.OutputConfirm2.setChecked(False)
            self.OutputConfirm2.setText("Confirm")
        else:
            self.Warning2.setText("")
        self.ModelOption.setEnabled(not self.OutputConfirm2.isChecked())
        self.OutputLoad2.setEnabled(not self.OutputConfirm2.isChecked())
        self.OutputLoad4.setEnabled(not self.OutputConfirm2.isChecked())

    def x_lock_clb(self):
        self.Xview.setChecked(True)
        self.Yview.setChecked(False)
        self.Zview.setChecked(False)
        self.glw.view = self.glw.camera.x_view()
        self.ViewIndex_setText_clb()

    def y_lock_clb(self):
        self.Yview.setChecked(True)
        self.Xview.setChecked(False)
        self.Zview.setChecked(False)
        self.glw.view = self.glw.camera.y_view()
        self.ViewIndex_setText_clb()

    def z_lock_clb(self):
        self.Zview.setChecked(True)
        self.Xview.setChecked(False)
        self.Yview.setChecked(False)
        self.glw.view = self.glw.camera.z_view()
        self.ViewIndex_setText_clb()

    def axis_clb(self):
        self.glw.vao["Axis"][-1] = not self.glw.vao["Axis"][-1]

    def camera_set(self):
        self.glw.view, n = self.glw.camera.set_view(self.ViewIndex.toPlainText())
        self.ViewIndex.setText(f"{n}")

    def copy_clb(self):
        self.clip_board.setImage(self.glw.grabFramebuffer())

    def save_clb(self):
        self.glw.grabFramebuffer().save(FileBrowser().create_file_name_dialog()[0], "png")

    # button trigger functions
    def interpolation_trigger(self):
        os.makedirs(f"{self.FilePath.text()}\\banira_files", exist_ok=True)

        self.TrainCase.hide()
        self.TrainCaseText.hide()
        self.bgwidget4.hide()
        self.TrainCase.clear()
        self.train_int = 0
        self.tr_case = """"""

        if self.model_signal == 0 and self.TrainRatio.text() == "":
            self.TrainRatio.setText("90")
            self.TestRatio.setText("10")

        self.DataLogBox.setText("")
        self.ProgressBar.set_value = 0
        self.Percentage.setText("")
        self.Confirm2.setChecked(False)
        self.Confirm3.setChecked(False)
        self.Confirm2.setText("Confirm")
        self.TestCase.setEnabled(True)
        self.TrainCase.setEnabled(True)
        self.TestCase.clear()
        self.test_int = 0
        case_runner = []
        case_train = []
        path = self.FilePath.text()

        def runner():
            train_signal = """"""
            runner_signal = """"""
            for item in case_train:
                train_signal += f"{item},"
            for item in case_runner:
                runner_signal += f"{item},"
            os.system(f"python .\\data_fitting\\interpolation_runner.pyw {self.FilePath.text()} {train_signal} {runner_signal}")

        if not self.Lock.isChecked():
            dlg = MyDialog(self)
            dlg.setWindowTitle("Warning")
            dlg.exec()
        else:
            for item in os.listdir(path):
                if os.path.isdir(path + '//' + item) and item != "banira_files":
                    case_train.append(item)
                    if not os.path.exists(f"{path}/banira_files/npy/{item}_xyz.npy"):
                        case_runner.append(item)

            if not case_runner:

                self.DataLogBox.setText("All finished")
                self.ProgressBar.set_value = 100
                self.Percentage.setText("100%")

                with open(f"{path}/banira_files/finished.txt", "w") as f:
                    for item in case_train:
                        f.write(f"{item}\n")
                    f.write("All finished\n")
                    f.close()

                self.TrainCase.show()
                self.TrainCaseText.show()
                self.bgwidget4.show()

                self.Start.setEnabled(True)

                for item in case_train:
                    if self.Model1Opt3.isChecked():
                        self.TrainCase.insertItem(self.train_int, item)
                        self.TestCase.insertItem(self.train_int, item)
                        self.train_int += 1
                        self.test_int += 1
                    else:
                        self.TrainCase.insertItem(self.train_int, item)
                        self.train_int += 1


                for i in range(self.train_int):
                    self.TrainCase.item(i).setSelected(True)

            elif self.Model1.isChecked():
                for item in ("counter.txt", "finished.txt", "log.txt"):
                    if os.path.exists(f"{path}/banira_files/" + item):
                        os.remove(f"{path}/banira_files/" + item)

                t1 = Thread(target=runner)
                t1.setDaemon(True)
                t1.start()
                self.total_quantite = 0
                self.timer.start()

                for item in case_train:
                    self.total_quantite += sum(
                        [len(os.listdir(f"{path}//{item}/training/" + color)) for color in
                         os.listdir(f"{path}//{item}/training/")]
                    )
                fini = self.total_quantite
                for item in case_runner:
                    fini -= sum(
                        [len(os.listdir(f"{path}//{item}/training/" + color)) for color in
                         os.listdir(f"{path}//{item}/training/")]
                    )
                with open(f"{path}/banira_files/counter.txt", "a") as f:
                    f.write(f"{fini}\n")
                    f.close()

                self.Percentage.setText(f"{round(fini/self.total_quantite)}%")
                self.ProgressBar.set_value = round(fini/self.total_quantite)
                self.DataLogBox.setText("")
                self.Load.setEnabled(False)
                self.Confirm.setEnabled(False)
            elif self.Model2.isChecked() and self.ProgressBar.set_value != 100:
                ...

    def data_log_printer(self):
        buf = []
        cnt_reader = []
        try:
            with open(f"{self.FilePath.text()}/banira_files/counter.txt", "r", encoding="utf8") as cnt:
                cnt_reader.extend([int(item) for item in cnt])
                cnt.close()
            cnt_number = sum(cnt_reader)
            if round(cnt_number / self.total_quantite * 100) != 100:
                self.ProgressBar.set_value = round(cnt_number / self.total_quantite * 100)
                self.Percentage.setText(f"{round(cnt_number / self.total_quantite * 100)}%")
            else:
                self.ProgressBar.set_value = 100
                self.Percentage.setText("100%")
        except FileNotFoundError:
            pass
        try:
            with open(f"{self.FilePath.text()}/banira_files/log.txt", "r", encoding="utf8") as f:
                for step, row in enumerate(f):
                    buf.append(row)
                f.close()
            if re.findall(find_all_finished, self.DataLogBox.toPlainText()[-20:]):
                self.timer.stop()
                self.ProgressBar.set_value = 100
                self.Percentage.setText(f"100%")
                os.remove(f"{self.FilePath.text()}/banira_files/counter.txt")
                self.Load.setEnabled(True)
                self.Confirm.setEnabled(True)
                self.Start.setEnabled(True)

                buf_case = []
                with open(f"{self.FilePath.text()}/banira_files/finished.txt", "r") as f:
                    for row in f:
                        if self.Model1Opt3.isChecked():
                            if not re.findall(find_all_finished, row):
                                self.TrainCase.insertItem(self.train_int, row[:-1])
                                self.TestCase.insertItem(self.train_int, row[:-1])
                                self.train_int += 1
                                self.test_int += 1
                            else:
                                break
                        else:
                            if not re.findall(find_all_finished, row):
                                self.TrainCase.insertItem(self.train_int, row[:-1])
                                self.train_int += 1
                            else:
                                break
                    f.close()
                for i in range(self.train_int):
                    self.TrainCase.item(i).setSelected(True)

                self.TrainCase.show()
                self.TrainCaseText.show()
                self.bgwidget4.show()

            else:
                if self.DataLogBox.toPlainText() == "":
                    for row in buf[:len(buf)]:
                        self.DataLogBox.append(row[:-1])
                else:
                    for row in buf[len(self.DataLogBox.toPlainText().split("\n")):len(buf)]:
                        self.DataLogBox.append(row[:-1])
        except FileNotFoundError:
            pass

    def discrate_classifier_trigger(self):

        self.PS_signal = 0

        self.FilePath.setEnabled(False)
        self.Load.setEnabled(False)
        self.Confirm.setEnabled(False)

        self.TrainRatio.setReadOnly(True)
        self.CaseOption.clear()
        self.case_path = []
        self.case_int = 0
        self.ModelOption.clear()
        self.model_path = []
        self.model_int = 0

        self.tabWidget.setTabEnabled(3, False)
        self.customized_graph.clear()
        self.customized_graph.hide()
        self.TrainingLogBox.setText("")
        self.ProgressNoticer.setText("")
        self.ImpleNoticer.setText("Predict data preparing...")

        if "pause" in os.listdir(f"{self.FilePath.text()}/banira_files"):
            os.remove(f"{self.FilePath.text()}/banira_files\\pause")
        if "stop" in os.listdir(f"{self.FilePath.text()}/banira_files"):
            os.remove(f"{self.FilePath.text()}/banira_files\\stop")

        self.tr_order = 0
        self.ex_order = 0

        self.timer4.start()

        case_pre_fin = []
        case_pre_run = []
        case = """"""
        def model_runner():
            os.system(
                f"python .\\training_process\\DiscrateClassifier.py {self.FilePath.text()} {self.EpochValue.text()} {self.TestRatio.text()} {self.model_signal} {self.test_case} {self.tr_case[:-1]}")

        def predict_file_runner():
            os.system(f"python .\\training_process\\preproceed_runner.pyw {self.FilePath.text()} {case}")

        for item in os.listdir(self.FilePath.text()):
            if os.path.isdir(self.FilePath.text() + '//' + item) and item != "banira_files":
                if not os.path.exists(f"{self.FilePath.text()}/banira_files/npy/{item}_xyz_predict.npy"):
                    case_pre_run.append(item)

        for item in os.listdir(f"{self.FilePath.text()}/banira_files/npy"):
            if item[-16:] == "_xyz_predict.npy":
                case_pre_fin.append(item[:-16])
        with open(f"{self.FilePath.text()}/banira_files/finished_ex.txt", "w") as f:
            for item in case_pre_fin:
                f.write(f"{item}\n")
            f.close()

        if not case_pre_run:
            self.ex_order = 1
            with open(f"{self.FilePath.text()}/banira_files/finished_ex.txt", "a") as f:
                f.write("All finished\n")

        for item in os.listdir(f"{self.FilePath.text()}\\banira_files"):
            if item in ("graph.txt", "training.txt", "time.txt"):
                os.remove(self.FilePath.text() + "/banira_files/" + item)


        try:
            float(self.EpochValue.text())
            if float(self.EpochValue.text()) % 1 > 1e-10 or float(self.EpochValue.text()) <= 0:
                dlg = MyDialog2(self, "Please enter positive integer number.")
                dlg.setWindowTitle("Warning")
                dlg.exec()
            else:
                if self.Model1.isChecked():
                    if self.tr_order == 0:  # TODO: comment
                        self.Start.setEnabled(False)
                        self.EpochValue.setEnabled(False)
                        t1 = Thread(target=model_runner)
                        t1.setDaemon(True)
                        t1.start()
                        self.timer2.start()
                        self.timer3.start()
                        self.legend.clear()
                        self.label_signal = 0
                    else:
                        self.ProgressNoticer.setText("Model training finished")
                        self.EpochValue.setEnabled(True)
                    if self.ex_order == 0:
                        for item in case_pre_run:
                            case += f"{item},"
                        case = case[:-1]
                        t2 = Thread(target=predict_file_runner)
                        t2.setDaemon(True)
                        t2.start()
                    else:
                        self.ImpleNoticer.setText("Predict data prepared")
                elif self.Model2.isChecked():
                    ...
        except ValueError:
            dlg = MyDialog2(self, "Please enter positive integer number.")
            dlg.setWindowTitle("Warning")
            dlg.exec()

    def training_log_printer(self):
        buf = []
        try:
            with open(f"{self.FilePath.text()}/banira_files/time.txt", "r") as f:
                box = np.loadtxt(f)
                f.close()
                assert box.nbytes >= 4
                self.ProgressNoticer.setText("Time spent: {}  Finish in: {}".format(timedelta(seconds=box[-2]),
                                                                                    "-:--:--" if np.isnan(
                                                                                        box[-1]) else timedelta(
                                                                                        seconds=box[-1])))
        except FileNotFoundError:
            pass
        except AssertionError:
            pass
        except ValueError:
            pass
        try:
            with open(f"{self.FilePath.text()}/banira_files/training.txt", "r", encoding="utf8") as f:
                for step, row in enumerate(f):
                    buf.append(row)
                f.close()
            if self.PS_signal == 0:
                self.Pause.setEnabled(True)
                self.Stop.setEnabled(True)
                self.PS_signal = 1

            if re.findall(find_finished_training, self.TrainingLogBox.toPlainText()[-100:]) or re.findall(find_stopped_training, self.TrainingLogBox.toPlainText()[-100:]):
                self.tr_order = 1
                self.timer2.stop()
                self.training_graph_printer()
                self.timer3.stop()
                self.EpochValue.setEnabled(True)
                self.Pause.setEnabled(False)
                self.Stop.setChecked(False)
            else:
                if self.TrainingLogBox.toPlainText() == "":
                    for row in buf[:len(buf)]:
                        self.TrainingLogBox.append(row[:-1])
                else:
                    for row in buf[len(self.TrainingLogBox.toPlainText().split("\n")):len(buf)]:
                        self.TrainingLogBox.append(row[:-1])
        except FileNotFoundError:
            pass

    def training_graph_printer(self):
        self.customized_graph.clear()
        try:
            with open(f"{self.FilePath.text()}/banira_files/graph.txt", "r") as f:
                box = np.loadtxt(f)
                f.close()
            x = []
            y = []
            z = []
            for step, item in enumerate(box):
                if step % 3 == 0:
                    x.append(item)
                elif step % 3 == 1:
                    y.append(item)
                else:
                    z.append(item)

            self.customized_graph.show()
            if x[-1] <= 5:
                self.customized_graph.setXRange(1, 5)
            elif x[-1] > 20:
                self.customized_graph.setXRange(x[-1] - 19, x[-1])
                self.customized_graph.setYRange(min(*y[-20:], *z[-20:]), max(*y[-20:], *z[-20:]))
            else:
                self.customized_graph.setXRange(1, x[-1])
            pen1 = pg.mkPen(color=(200, 111, 99), width=2)
            pen2 = pg.mkPen(color=(200, 145, 45), width=2)
            l1 = self.customized_graph.plot(x, y, pen=pen1, symbol='o', symbolSize=5, symbolBrush=(160, 87, 77))
            l2 = self.customized_graph.plot(x, z, pen=pen2, symbol='o', symbolSize=5, symbolBrush=(185, 130, 45))
            if self.label_signal == 0:
                fontCssLegend = '<style type="text/css"> p {font-family: Dubai; font-size: 10.5pt; color: "#4e526b"} </style>'
                self.legend.addItem(l1, name=fontCssLegend + '<p>Training loss</p>')
                self.legend.addItem(l2, name=fontCssLegend + '<p>Validation loss</p>')
                self.label_signal += 1
        except FileNotFoundError:
            pass
        except AssertionError:
            pass
        except ValueError:
            pass

    def training_finish_order(self):
        buf_ex = """"""
        try:
            with open(f"{self.FilePath.text()}\\banira_files\\finished_ex.txt", "r") as f:
                for row in f:
                    buf_ex += row
                f.close()
            if re.findall(find_all_finished, buf_ex[-13:]):
                self.ex_order = 1
                self.ImpleNoticer.setText("Predict data prepared")
        except FileNotFoundError:
            pass
        if self.tr_order == 1 and self.ex_order == 1:
            for step, row in enumerate(buf_ex.split("\n")[:-2]):
                self.CaseOption.insertItem(self.case_int, row)
                self.case_path.append(f"{self.FilePath.text()}\\banira_files\\npy\\{row}_xyz_predict.npy")
                self.case_int += 1
            for item in os.listdir(f"{self.FilePath.text()}\\banira_files\\TumourClassifier"):
                self.ModelOption.insertItem(self.model_int, item)
                self.model_path.append(f"{self.FilePath.text()}\\banira_files\\TumourClassifier\\{item}")
                self.model_int += 1
            self.timer4.stop()
            self.TrainRatio.setReadOnly(False)
            self.FilePath.setEnabled(True)
            self.Load.setEnabled(True)
            self.Confirm.setEnabled(True)
            self.tabWidget.setTabEnabled(3, True)
            self.ImpleNoticer.setText("Implementation available")
            self.Start.setEnabled(True)
            self.Pause.setEnabled(False)
            self.Stop.setEnabled(False)

    def predict_trigger(self):
        def predict_runner():
            os.system(
                f"python .\\implementation\\predict.pyw {self.model_path[self.ModelOption.currentRow()]} {self.case_path[self.CaseOption.currentRow()]} {self.CaseOption.currentItem().text() + '_' + self.ModelOption.currentItem().text()}")

        if not (self.OutputConfirm.isChecked() == True and self.OutputConfirm2.isChecked() == True):
            dlg = MyDialog2(self, "Please confirm case and model.")
            dlg.setWindowTitle("Warning")
            dlg.exec()
        else:
            self.tabWidget.setTabEnabled(4, False)
            self.Compute.setEnabled(False)
            try:
                assert f"ai__{self.CaseOption.currentItem().text()}_{self.ModelOption.currentItem().text()}.csv" in os.listdir(
                    ".//output_csv")
                self.Finished.setText("Existed")
                self.ViewPortLog.setText(
                    f"Case name:\n{self.CaseOption.currentItem().text()}\nModel implied:\n{self.ModelOption.currentItem().text()}\n")
                self.tabWidget.setTabEnabled(4, True)
                self.Compute.setEnabled(True)
                from visualization.loader import Classifier
                buffer = Classifier()(
                    f".//output_csv//ai__{self.CaseOption.currentItem().text()}_{self.ModelOption.currentItem().text()}.csv")
                buffer[:, 0] -= np.average(buffer[:, 0])
                buffer[:, 1] -= np.average(buffer[:, 1])
                buffer[:, 2] -= np.average(buffer[:, 2])
                self.glw.load_buffer("ai__", buffer, [[0, buffer.shape[0]]])

            except AssertionError:
                t_predict = Thread(target=predict_runner)
                t_predict.setDaemon(True)
                t_predict.start()
                self.Finished.setText("Processing...")
                self.timer5.start(10000)

    def predict_finish_order(self):
        try:
            assert f"ai__{self.CaseOption.currentItem().text()}_{self.ModelOption.currentItem().text()}.csv" in os.listdir(
                ".//output_csv")
            time.sleep(1.5)  # todo: buggy
            self.timer5.stop()
            from visualization.loader import Classifier
            buffer = Classifier()(
                f".//output_csv//ai__{self.CaseOption.currentItem().text()}_{self.ModelOption.currentItem().text()}.csv")
            buffer[:, 0] -= np.average(buffer[:, 0])
            buffer[:, 1] -= np.average(buffer[:, 1])
            buffer[:, 2] -= np.average(buffer[:, 2])
            self.glw.load_buffer("ai__", buffer, [[0, buffer.shape[0]]])
            self.Finished.setText("Finished")
            self.ViewPortLog.setText(
                f"Case name:\n{self.CaseOption.currentItem().text()}\nModel implied:\n{self.ModelOption.currentItem().text()}\n")
            self.tabWidget.setTabEnabled(4, True)
            self.Compute.setEnabled(True)

        except AssertionError:
            pass


class MyDialog(QtWidgets.QDialog, Ui_WarningWindow):
    def __init__(self, father):
        super(MyDialog, self).__init__()
        my_dialog = self.setupUi(self)
        self.OKButton.clicked.connect(lambda event: self.warning_reaction())
        self.father = father

    def warning_reaction(self):
        self.father.tabWidget.setCurrentIndex(0)
        self.close()


class MyDialog2(QtWidgets.QDialog, Ui_WarningWindow):
    def __init__(self, father, text):
        super(MyDialog2, self).__init__()
        my_dialog = self.setupUi(self)
        self.OKButton.clicked.connect(lambda event: self.close())
        self.WarningText.setText(text)


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

    def create_file_name_dialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name = QtWidgets.QFileDialog.getSaveFileName(self, '', "untitled.png", "Images(*.png *.jpg)", options=options)
        return file_name

class MyGLWidget(QOpenGLWidget):
    def __init__(self, real_dad, dad):
        super().__init__(real_dad)
        self.dad = dad
        self.bg = [1.0, 1.0, 1.0, 1.0]
        # refresh rate
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.update)
        self.timer.start()
        self.start_time = time.time()
        self.setMouseTracking(True)
        self.freeze = False

        self.w = 621
        self.h = 591
        # VAO and Draw-list
        self.vao = {}
        self.display_list = []
        self.makeCurrent()

        import visualization.camera as camera
        self.camera = camera.Camera()

    def load_buffer(self, tag: str, buffer: np.ndarray, pointer: list):
        self.makeCurrent()
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, buffer.nbytes, buffer, GL_STATIC_DRAW)
        # vertex shape: [x, y, z, r, g, b]
        if buffer.shape[1] == 6:
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        # TODO: this will be removed or improved.
        elif buffer.shape[1] == 3:
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        elif buffer.shape[1] == 8:
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        self.vao[tag] = [vao, pointer, True]

    def initializeGL(self):
        from visualization import shader_src
        self.shader = compileProgram(compileShader(shader_src.vertex_src, GL_VERTEX_SHADER),
                                     compileShader(shader_src.fragment_src, GL_FRAGMENT_SHADER))
        glUseProgram(self.shader)
        glClearColor(*self.bg)
        glEnable(GL_DEPTH_TEST)

        self.projection = pyrr.matrix44.create_perspective_projection(45, 621 / 591, 0.001, 1000)
        self.view = pyrr.matrix44.create_look_at(pyrr.Vector3([0.0, 0.0, 30.0]), pyrr.Vector3([0.0, 0.0, -1.0]),
                                                 pyrr.Vector3([0, 1, 0]))
        model = pyrr.matrix44.create_from_z_rotation(0.0)

        self.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.view_loc = glGetUniformLocation(self.shader, "view")
        model_loc = glGetUniformLocation(self.shader, "model")
        self.color_loc = glGetUniformLocation(self.shader, "color")

        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.projection)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.view)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)



        glPointSize(4)

        self.current_frame = 0
        self.color_flag = 0
        self.color_pos = glGetUniformLocation(self.shader, "color_flag")
        # load vao
        import visualization.Coordinates as Coordinates
        axis = Coordinates.Coord()
        self.vao["Axis"] = [axis.vao, [[0, axis.indices.nbytes // 4]], True]

        self.color_loc = glGetUniformLocation(self.shader, "color")

        vertex_src = """"""
        fragment_src = """"""
        with open(r"./visualization/shaders/vertex_shader.shader", "r") as f:
            for row in f:
                vertex_src += row
            f.close()
        with open(r"./visualization/shaders/fragment_shader.shader", "r") as f:
            for row in f:
                fragment_src += row
            f.close()
        self.ai_shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                        compileShader(fragment_src, GL_FRAGMENT_SHADER))
        glUseProgram(self.ai_shader)
        self.ai_proj_loc = glGetUniformLocation(self.ai_shader, "projection")
        self.ai_view_loc = glGetUniformLocation(self.ai_shader, "view")
        ai_model_loc = glGetUniformLocation(self.ai_shader, "model")
        glUniformMatrix4fv(self.ai_proj_loc, 1, GL_FALSE, self.projection)
        glUniformMatrix4fv(self.ai_view_loc, 1, GL_FALSE, self.view)
        glUniformMatrix4fv(ai_model_loc, 1, GL_FALSE, model)
        glUseProgram(self.shader)

    def resizeGL(self, w, h):
        self.w = w
        self.h = h
        glViewport(0, 0, w, h)
        self.projection = pyrr.matrix44.create_perspective_projection(45, w / h, 0.001, 1000)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.projection)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.view)
        for key in self.vao.keys():
            vao, pointer, draw = self.vao[key]
            if draw and key != "Axis" and key[:4] != "ai__":
                glUniform1i(self.color_pos, 1)
                glBindVertexArray(vao)
                # frame by frame
                glDrawArrays(GL_POINTS,
                             pointer[self.current_frame % len(pointer)][0],
                             pointer[self.current_frame % len(pointer)][1] - pointer[self.current_frame % len(pointer)][
                                 0])
            elif draw and key == "Axis":
                glUniform1i(self.color_pos, 0)
                glBindVertexArray(vao)
                glDrawElements(GL_TRIANGLES, pointer[0][1], GL_UNSIGNED_INT, None)
            elif draw and key[:4] == "ai__":
                glUseProgram(self.ai_shader)
                glUniformMatrix4fv(self.ai_proj_loc, 1, GL_FALSE, self.projection)
                glUniformMatrix4fv(self.ai_view_loc, 1, GL_FALSE, self.view)
                glBindVertexArray(vao)
                # frame by frame
                glDrawArrays(GL_POINTS,
                             pointer[self.current_frame % len(pointer)][0],
                             pointer[self.current_frame % len(pointer)][1] - pointer[self.current_frame % len(pointer)][
                                 0])
                glUseProgram(self.shader)

        if not self.freeze:
            self.current_frame += 1
        self.current_frame %= 900

    def mouseMoveEvent(self, a0):
        x_pos = a0.pos().x()
        y_pos = a0.pos().y()
        if abs(x_pos) < 400 and abs(y_pos) < 400:
            self.grabKeyboard()
        else:
            self.releaseKeyboard()
        x_pos -= self.size().width() / 2
        y_pos *= -1
        y_pos += self.size().height() / 2
        if self.camera.mouse_left:
            # x, y, z view detached
            self.dad.Xview.setChecked(False)
            self.dad.Yview.setChecked(False)
            self.dad.Zview.setChecked(False)

            dx = self.camera.mouse_pos.x - x_pos
            dy = self.camera.mouse_pos.y - y_pos
            self.camera.mouse_pos = pyrr.Vector3([x_pos, y_pos, 0.0])
            delta = pyrr.Vector3([dx, dy, np.sqrt(dx ** 2 + dy ** 2) / 100])
            self.view = self.camera(delta, flag="left")

            self.dad.ViewIndex_setText_clb()

        elif self.camera.mouse_middle:
            dx = self.camera.mouse_pos.x - x_pos
            dy = self.camera.mouse_pos.y - y_pos
            self.camera.mouse_pos = pyrr.Vector3([x_pos, y_pos, 0.0])
            delta = pyrr.Vector3([dx, dy, 0])
            self.view = self.camera(delta / 5, flag="middle")

            self.dad.ViewIndex_setText_clb()
        else:
            self.camera.mouse_pos = pyrr.Vector3([x_pos, y_pos, 0.0])

    def mousePressEvent(self, a0):
        if a0.button() == 1:  # 1==left, 4=middle
            self.camera.mouse_left = True
        if a0.button() == 4:  # 1==left, 4=middle
            self.camera.mouse_middle = True

    def mouseReleaseEvent(self, a0):
        if a0.button() == 1:  # 1==left, 4=middle
            self.camera.mouse_left = False
        if a0.button() == 4:  # 1==left, 4=middle
            self.camera.mouse_middle = False

    def wheelEvent(self, a0) -> None:
        y_offset = a0.angleDelta().y() / 120

        def f():
            if sum([abs(item) for item in self.camera.position.xyz]) <= 1.01:
                if y_offset >= 0:
                    return
            for i in range(50):
                self.camera.position += self.camera.front * y_offset * 0.02
                self.camera.position = pyrr.Vector4([*self.camera.position.xyz, 1.0])
                self.view = self.camera(flag="wheel")

                time.sleep(0.005)
                if abs(sum([*self.camera.position.xyz])) <= 1.01:
                    if y_offset >= 0:
                        return

        t = Thread(target=f)
        t.start()

        self.dad.ViewIndex_setText_clb()


def main():
    app = QtWidgets.QApplication(sys.argv)
    clipboard = app.clipboard()
    window = MainWindow(clipboard)
    window.show()
    window.tabWidget.setCurrentIndex(4)
    window.tabWidget.setCurrentIndex(0)
    sys.exit(app.exec_())


if __name__ == '__main__':
    QtWidgets.QTabWidget = MyTabWidget
    main()
