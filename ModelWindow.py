from PyQt5.QtWidgets import QSlider, QLabel, QMessageBox, QCheckBox, QHBoxLayout
from PyQt5.QtWidgets import QListWidget, QWidget, QPushButton, QDialog, QVBoxLayout
from PyQt5 import QtCore, QtWidgets, QtGui
from tensorflow import keras
import shutil
import os

class ModelWindow(QDialog):
    def __init__(self):
        super(ModelWindow, self).__init__()
        self.setWindowTitle("Model selection")
        self.model = None
        self.name = None
        mainLayout = QVBoxLayout(self)
        hWidget = QWidget()
        hLayout = QHBoxLayout(hWidget)
        text = QLabel("Select a neural network model : ")
        list = QListWidget()
        self.select = QPushButton('select')
        self.select.clicked.connect(lambda :self.ok_pressed(list.currentItem().text()))
        self.delete = QPushButton('delete')
        self.delete.clicked.connect(lambda :self.delete_pressed(list))
        cancel = QPushButton('cancel')
        cancel.clicked.connect(self.cancel_pressed)

        dir = [name for name in os.listdir(".") if os.path.isdir(name) and name.startswith("model_")]
        if len(dir)>0: list.addItems(['_'.join(name.split('_')[1:]) for name in dir])
        else: self.checkCount(list)

        mainLayout.addWidget(text)
        mainLayout.addWidget(list)
        hLayout.addWidget(cancel)
        hLayout.addWidget(self.delete)
        hLayout.addWidget(self.select)
        hLayout.setContentsMargins(0,0,0,0)
        mainLayout.addWidget(hWidget)
        self.setLayout(mainLayout)


    def checkCount(self, list):
        if list.count() == 0:
            list.addItems(["No models found"])
            list.setEnabled(False)
            self.select.setEnabled(False)
            self.delete.setEnabled(False)

    def getModel(self):
        return(self.name, self.model)


    def ok_pressed(self, selected):
        print(selected, 'selected')
        try:
            self.model = keras.models.load_model('model_'+selected)
            self.name = selected
        except:
            print('Cannot load model')
        self.accept()


    def delete_pressed(self, list):
        shutil.rmtree('model_'+list.currentItem().text())
        list.takeItem(list.currentRow())
        self.checkCount(list)


    def cancel_pressed(self):
        self.reject()
