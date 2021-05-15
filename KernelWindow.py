from PyQt5.QtWidgets import QSlider, QLabel, QMessageBox, QCheckBox, QHBoxLayout
from PyQt5.QtWidgets import QListWidget, QWidget, QPushButton, QDialog, QVBoxLayout
from PyQt5 import QtCore, QtWidgets, QtGui
from tensorflow import keras
import shutil
import os

class KernelWindow(QDialog):
    def __init__(self):
        super(KernelWindow, self).__init__()
        self.setWindowTitle("Kernel selection")
        self.model = None
        self.name = None
        mainLayout = QVBoxLayout(self)
        hWidget = QWidget()
        hLayout = QHBoxLayout(hWidget)
        text = QLabel("Select a kernel for the convolution : ")
        list = QListWidget()

        dir = ["identity", "sharpen", "blur", "bottom sobel", "emboss kernel", "left sobel", "outline", "right sobel", "top sobel"]
        if len(dir)>0:
            list.addItems([name for name in dir])
        else: self.checkCount(list)

        mainLayout.addWidget(text)
        mainLayout.addWidget(list)

        hWidgetConv = QWidget()
        hLayoutConv = QHBoxLayout(hWidgetConv)
        textConv = QLabel("Select a number of convolutional layers : ")
        listConv = QListWidget()
        dirConv = [str(i) for i in range(1, 6)]
        if len(dirConv)>0:
            listConv.addItems([name for name in dirConv])
        else: self.checkCount(listConv)

        self.select = QPushButton('select')
        self.select.clicked.connect(lambda :self.ok_pressed(list.currentItem().text(), listConv.currentItem().text()))
        cancel = QPushButton('cancel')
        cancel.clicked.connect(self.cancel_pressed)

        mainLayout.addWidget(textConv)
        mainLayout.addWidget(listConv)
        hLayoutConv.setContentsMargins(0,0,0,0)
        mainLayout.addWidget(hWidgetConv)


        hLayout.addWidget(cancel)
        hLayout.addWidget(self.select)
        hLayout.setContentsMargins(0,0,0,0)
        mainLayout.addWidget(hWidget)

        self.setLayout(mainLayout)


    def checkCount(self, list):
        if list.count() == 0:
            list.addItems(["No kernels found"])
            list.setEnabled(False)
            self.select.setEnabled(False)
            self.delete.setEnabled(False)

    def getKernel(self):
        return(self.kernel_name)

    def getNlayers(self):
        return (self.n_layers)

    def ok_pressed(self, selectedKernel, selectedN):
        print(selectedKernel, 'kernel selected')
        print(selectedN, 'layers selected')
        try:
            self.kernel_name = selectedKernel
            self.n_layers = selectedN
        except:
            print('Cannot choose this kernel')
        self.accept()

    def cancel_pressed(self):
        self.reject()

