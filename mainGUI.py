"""
Author: Olivier Renson
"""

import sys
from PyQt5 import QtGui, QtCore
from MainWindow import MyWindow
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())
