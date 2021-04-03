from PyQt5.QtWidgets import QTabWidget, QLabel, QMessageBox, QFileDialog, QHBoxLayout
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QMessageBox, QVBoxLayout
from PyQt5 import QtCore, QtWidgets, QtGui
from ModelWindow import ModelWindow
from skimage import io, transform
import pandas as pd

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setWindowTitle("Dog & Cat classifier")
        self.setFixedSize(600, 500)
        centralwidget = QWidget(self)
        tab = QTabWidget(centralwidget)
        tab.setGeometry(0, 0, 600, 500)
        tab.addTab(PredictTab(), "Predictor")
        tab.addTab(CNNTab(), "Home made CNN")
        tab.addTab(CNNTab(), "Resnet 10")
        self.setCentralWidget(centralwidget)


class PredictTab(QWidget):
    def __init__(self):
        super(PredictTab, self).__init__()
        self.imgPath = []
        self.imgIndex = 0
        self.predictions = []
        self.cnn = None
        mainLayout = QVBoxLayout(self)

        self.imgLabel = QLabel()
        self.imgLabel.setStyleSheet("background-color: lightgrey; border: 1px solid gray;")
        self.imgLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.prevButton = QPushButton("<")
        self.prevButton.setMaximumWidth(50)
        self.prevButton.setEnabled(False)
        self.nextButton = QPushButton(">")
        self.nextButton.setMaximumWidth(50)
        self.nextButton.setEnabled(False)
        self.predLabel = QLabel("None")
        self.predLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.predLabel.setFixedWidth(100)
        self.predLabel.setFixedHeight(20)
        hWidget1 = QWidget(self)
        hWidget1.setFixedHeight(20)
        hLayout1 = QHBoxLayout(hWidget1)
        hLayout1.setContentsMargins(0,0,0,0)
        hWidget2 = QWidget(self)
        hWidget2.setFixedHeight(25)
        hLayout2 = QHBoxLayout(hWidget2)
        hLayout2.setContentsMargins(0,0,0,0)
        hWidget3 = QWidget(self)
        hWidget3.setFixedHeight(25)
        hLayout3 = QHBoxLayout(hWidget3)
        hLayout3.setContentsMargins(0,0,0,0)
        #hWidget.setStyleSheet("border: 1px solid red; padding: 0 0 0 0; margin: 0px;")

        loadButton = QPushButton("Select picture(s)")
        modelButton = QPushButton("Select model (none)")
        predButton = QPushButton("Predict")
        exportButton = QPushButton("Export")
        loadButton.clicked.connect(self.loadImg)
        self.prevButton.clicked.connect(self.prevImg)
        self.nextButton.clicked.connect(self.nextImg)
        modelButton.clicked.connect(lambda :self.selectedModel(modelButton))
        predButton.clicked.connect(self.predict)
        exportButton.clicked.connect(self.export)

        mainLayout.addWidget(self.imgLabel)
        hLayout1.addWidget(self.prevButton)
        hLayout1.addWidget(self.predLabel)
        hLayout1.addWidget(self.nextButton)
        hLayout2.addWidget(loadButton)
        hLayout2.addWidget(modelButton)
        hLayout3.addWidget(predButton)
        hLayout3.addWidget(exportButton)
        mainLayout.addWidget(hWidget1)
        mainLayout.addWidget(hWidget2)
        mainLayout.addWidget(hWidget3)

    def loadImg(self):
        dialog = QFileDialog()
        dialog.setWindowTitle("Choose an image")
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setViewMode(QFileDialog.Detail)

        if dialog.exec_():
            self.imgPath = [str(i) for i in dialog.selectedFiles()]
            self.predictions = [None for i in range(len(self.imgPath))]
            self.imgIndex = 0
            print('Selection :')
            for i in self.imgPath: print(i)
            self.prevButton.setEnabled(False)
            if len(self.imgPath) >1: self.nextButton.setEnabled(True)
            elif len(self.imgPath) ==1: self.nextButton.setEnabled(False)
            self.updatePixmap(self.imgPath[self.imgIndex])

    def updatePixmap(self, path):
        self.imgLabel.setPixmap(QtGui.QPixmap(path))
        #self.imgLabel.setScaledContents(True)
        if self.predictions[0] is not None:
            self.predLabel.setText(str(self.predictions[self.imgIndex]))

    def predict(self):
        if len(self.imgPath)>0 and self.cnn is not None:
            for i in range(len(self.imgPath)):
                img = transform.resize(io.imread(self.imgPath[i]), (256,256), anti_aliasing=True)
                self.predictions[i] = self.cnn.predict(img.reshape(1, 256, 256, 3), 1)[0][0]
                print(self.predictions[i])
            self.updatePixmap(self.imgPath[self.imgIndex])

        else:
            QMessageBox(QMessageBox.Warning, "Error",\
                "Please select images and notwork model before making prediction").exec_()


    def nextImg(self):
        self.imgIndex +=1
        self.updatePixmap(self.imgPath[self.imgIndex])
        if self.imgIndex == len(self.imgPath)-1:
            self.nextButton.setEnabled(False)
        else:
            self.nextButton.setEnabled(True)
        self.prevButton.setEnabled(True)

    def prevImg(self):
        self.imgIndex -=1
        self.updatePixmap(self.imgPath[self.imgIndex])
        if self.imgIndex == 0:
            self.prevButton.setEnabled(False)
        else:
            self.prevButton.setEnabled(True)
        self.nextButton.setEnabled(True)

    def selectedModel(self, btn):
        win = ModelWindow()
        if win.exec_():
            name, self.cnn = win.getModel()
            btn.setText('Select model ({})'.format(name))

    def export(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fname, _ = QFileDialog.getSaveFileName(self,"Save as","Export.csv","All (.csv)", options=options)

        if fname:
            data_tuples = list(zip(self.imgPath,self.predictions))
            df = pd.DataFrame(data_tuples, columns=['Images','Predictions'])
            df.to_csv(fname)
            print(fname, "saved")


class CNNTab(QWidget):
    def __init__(self):
        super(CNNTab, self).__init__()
        mainLayout = QVBoxLayout(self)
        trainButton = QPushButton("load", self)

        #mainLayout.addWidget(trainButton)
