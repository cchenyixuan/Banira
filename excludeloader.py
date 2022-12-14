# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'excludeloader.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ExcludeLoader(object):
    def setupUi(self, ExcludeLoader):
        ExcludeLoader.setObjectName("ExcludeLoader")
        ExcludeLoader.resize(640, 480)
        ExcludeLoader.setMinimumSize(QtCore.QSize(640, 480))
        ExcludeLoader.setMaximumSize(QtCore.QSize(640, 480))
        ExcludeLoader.setStyleSheet("background:rgb(212, 218, 236)")
        self.centralwidget = QtWidgets.QWidget(ExcludeLoader)
        self.centralwidget.setObjectName("centralwidget")
        self.Confirm = QtWidgets.QPushButton(self.centralwidget)
        self.Confirm.setGeometry(QtCore.QRect(490, 130, 101, 31))
        self.Confirm.setStyleSheet("QPushButton{\n"
"font:13pt \"Dubai\";\n"
"color:#4e526b;\n"
"background-color:#f6f5ff;\n"
"border-radius:10px;\n"
"border-style:solid;\n"
"border-width:2px;\n"
"border-color:#8186b0;\n"
"padding:5px;\n"
"}\n"
"\n"
"QPushButton::hover{\n"
"color:#ffffff;\n"
"background-color:#9fa7e7;\n"
"border-color:#8186b0;\n"
"}\n"
"\n"
"QPushButton::pressed,QPushButton::checked{\n"
"color:#ffffff;\n"
"background-color:qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #9099d1, stop:1 #b0bbff)\n"
"}\n"
"")
        self.Confirm.setObjectName("Confirm")
        self.FilePath = QtWidgets.QLineEdit(self.centralwidget)
        self.FilePath.setGeometry(QtCore.QRect(50, 80, 541, 31))
        self.FilePath.setStyleSheet("background-color:rgb(255, 255, 255);\n"
"font:13pt \"Dubai\";\n"
"border-radius:10px;\n"
"border-style:solid;\n"
"border-width:2px;\n"
"border-color:#8186b0;")
        self.FilePath.setObjectName("FilePath")
        self.Load = QtWidgets.QPushButton(self.centralwidget)
        self.Load.setGeometry(QtCore.QRect(370, 130, 101, 31))
        self.Load.setStyleSheet("QPushButton{\n"
"font:13pt \"Dubai\";\n"
"color:#4e526b;\n"
"background-color:#f6f5ff;\n"
"border-radius:10px;\n"
"border-style:solid;\n"
"border-width:2px;\n"
"border-color:#8186b0;\n"
"padding:5px;\n"
"}\n"
"\n"
"QPushButton::hover{\n"
"color:#ffffff;\n"
"background-color:#9fa7e7;\n"
"border-color:#8186b0;\n"
"}\n"
"\n"
"QPushButton::pressed,QPushButton::checked{\n"
"color:#ffffff;\n"
"background-color:qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #9099d1, stop:1 #b0bbff)\n"
"}\n"
"")
        self.Load.setCheckable(False)
        self.Load.setAutoRepeat(False)
        self.Load.setAutoDefault(False)
        self.Load.setDefault(False)
        self.Load.setFlat(False)
        self.Load.setObjectName("Load")
        self.ProgressBar = QtWidgets.QWidget(self.centralwidget)
        self.ProgressBar.setGeometry(QtCore.QRect(40, 180, 511, 41))
        self.ProgressBar.setObjectName("ProgressBar")
        self.Percentage = QtWidgets.QLabel(self.centralwidget)
        self.Percentage.setGeometry(QtCore.QRect(550, 180, 51, 41))
        self.Percentage.setStyleSheet("font:12pt \"Dubai\";\n"
"color:#4e526b;")
        self.Percentage.setObjectName("Percentage")
        ExcludeLoader.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(ExcludeLoader)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 640, 23))
        self.menubar.setObjectName("menubar")
        ExcludeLoader.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(ExcludeLoader)
        self.statusbar.setObjectName("statusbar")
        ExcludeLoader.setStatusBar(self.statusbar)

        self.retranslateUi(ExcludeLoader)
        QtCore.QMetaObject.connectSlotsByName(ExcludeLoader)

    def retranslateUi(self, ExcludeLoader):
        _translate = QtCore.QCoreApplication.translate
        ExcludeLoader.setWindowTitle(_translate("ExcludeLoader", "Exclude file processor"))
        self.Confirm.setText(_translate("ExcludeLoader", "Confirm"))
        self.Load.setText(_translate("ExcludeLoader", "Load"))
        self.Percentage.setText(_translate("ExcludeLoader", "TextLabel"))
