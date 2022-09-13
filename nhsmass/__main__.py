from .gui import App
from PyQt5 import QtWidgets
import sys
sys.path.append('..')

app = QtWidgets.QApplication(sys.argv)
window = App()
window.show()
app.exec_()