
from PyQt5.QtWidgets import QApplication,QMainWindow,QWidget,QPushButton
from PyQt5.QtCore import QThread,pyqtSignal,QBasicTimer
from ui import *
import sys


class Main_Window(QWidget,Ui_Form):
    def __init__(self,parent=None):
        super(Main_Window, self).__init__(parent)
        self.setupUi(self)

        self.btn.clicked.connect(self.btn_s)
        # self.timer=QBasicTimer()

    #
    # def timerEvent(self, a0: 'QTimerEvent') -> None:



    def btn_s(self):


class light(QPushButton):
    def __init__(self,text):







if __name__ =='__main__':
    app=QApplication(sys.argv)
    win=Main_Window()
    win.show()
    sys.exit(app.exec_())