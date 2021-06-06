from PyQt5.QtWidgets import QTabWidget, QLabel, QMessageBox, QFileDialog, QHBoxLayout
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QMessageBox, QVBoxLayout
from PyQt5 import QtCore, QtWidgets, QtGui
from ModelWindow import ModelWindow
from KernelWindow import KernelWindow
from skimage import io, transform
from Conv_operation import Transformations
import pandas as pd

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setWindowTitle("Dog & Cat classifier")
        self.setFixedSize(600, 500)
        centralwidget = QWidget(self)
        PredictTab(centralwidget)
        self.setCentralWidget(centralwidget)


class PredictTab(QWidget):
    def __init__(self, parent):
        super(PredictTab, self).__init__(parent)
        self.setFixedSize(600, 500)
        self.imgPath = []
        self.imgIndex = 0
        self.predictions = []
        self.cnn = None
        self.kernel_name = None
        self.n_layers = 1
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
        self.predLabel.setFixedWidth(300)
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
        hWidget4 = QWidget(self)
        hWidget4.setFixedHeight(25)
        hLayout4 = QHBoxLayout(hWidget4)
        hLayout4.setContentsMargins(0,0,0,0)
        #hWidget.setStyleSheet("border: 1px solid red; padding: 0 0 0 0; margin: 0px;")

        loadButton = QPushButton("Select picture(s)")
        modelButton = QPushButton("Select model (none)")
        predButton = QPushButton("Predict")
        exportButton = QPushButton("Export")
        convolveButton = QPushButton("Apply convolution")
        kernelButton = QPushButton("Choose kernel")
        loadButton.clicked.connect(self.loadImg)
        self.prevButton.clicked.connect(self.prevImg)
        self.nextButton.clicked.connect(self.nextImg)
        modelButton.clicked.connect(lambda :self.selectedModel(modelButton))
        predButton.clicked.connect(self.predict)
        exportButton.clicked.connect(self.export)
        kernelButton.clicked.connect(lambda :self.choose_kernel(kernelButton))
        convolveButton.clicked.connect(self.convolve)

        mainLayout.addWidget(self.imgLabel)
        hLayout1.addWidget(self.prevButton)
        hLayout1.addWidget(self.predLabel)
        hLayout1.addWidget(self.nextButton)
        hLayout2.addWidget(loadButton)
        hLayout2.addWidget(modelButton)
        hLayout3.addWidget(predButton)
        hLayout3.addWidget(exportButton)
        hLayout4.addWidget(convolveButton)
        hLayout4.addWidget(kernelButton)
        mainLayout.addWidget(hWidget1)
        mainLayout.addWidget(hWidget2)
        mainLayout.addWidget(hWidget3)
        mainLayout.addWidget(hWidget4)

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
            self.updatePixmap(self.imgPath[self.imgIndex])
            self.prevButton.setEnabled(False)
            if len(self.imgPath) >1: self.nextButton.setEnabled(True)
            elif len(self.imgPath) ==1: self.nextButton.setEnabled(False)
            self.updatePixmap(self.imgPath[self.imgIndex])
            if self.cnn is not None: self.predict()

    def updatePixmap(self, path):
        self.imgLabel.setPixmap(QtGui.QPixmap(path).scaled(500, 500))
        self.predLabel.setText(str(self.predictions[self.imgIndex]))
        if self.predictions[self.imgIndex] is None:
            self.predLabel.setText("I don't know yet")
        elif self.predictions[self.imgIndex] < 0.5:
            p = str(round((1-self.predictions[self.imgIndex])*100,2))
            self.predLabel.setText("I think it's a cat ! "+p+"%")
        elif self.predictions[self.imgIndex] > 0.5:
            p = str(round(self.predictions[self.imgIndex]*100,2))
            self.predLabel.setText("I think it's a dog ! "+p+"%")

    def predict(self):
        if len(self.imgPath)>0 and self.cnn is not None:
            for i in range(len(self.imgPath)):
                img = transform.resize(io.imread(self.imgPath[i]), (256,256), anti_aliasing=True)
                self.predictions[i] = self.cnn.predict(img.reshape(1, 256, 256, 3), 1)[0][0]
                print(self.predictions[i])
            self.updatePixmap(self.imgPath[self.imgIndex])

        else:
            QMessageBox(QMessageBox.Warning, "Error",\
                "Please select images and neural notwork model before making prediction").exec_()


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

    def choose_kernel(self, btn):
        win = KernelWindow()
        if win.exec_():
            self.kernel_name = win.getKernel()
            self.n_layers = int(win.getNlayers())
            btn.setText('Select kernel ({})'.format(self.kernel_name))

    def convolve(self):
        if self.kernel_name == None:
            QMessageBox(QMessageBox.Warning, "Error",\
                "Please select a kernel before convoluting").exec_()
        else:
            for img in self.imgPath:
                # Apply one convolution + pooling operation
                t = Transformations(img)
                t.choose_kernel(self.kernel_name)
                conv_name = t.multilayer(self.n_layers)
                self.imgLabel.setPixmap(QtGui.QPixmap(conv_name).scaled(500, 500))
                self.predLabel.setText(str(self.kernel_name + " convolution with " + str(self.n_layers) + " layers and maxpooling"))

                
                
