from PyQt5.QtWidgets import QFileDialog,QApplication,QWidget
from PyQt5.QtCore import QThread,pyqtSignal
from predict.image_predict import detect
from yolo_net.Class_Yolo import YOLO
from winsound import Beep
import numpy as np
import cv2
from ui_utils import *
from PyQt5.QtGui import *
import os
from new_ui import Ui_Form
import sys
from time import time

temp_path=r"../predict/res/cache/temp.jpg"

class window(QWidget,Ui_Form):
    def __init__(self,parent=None):
        super(window, self).__init__(parent)
        self.setupUi(self)
        self.print.append('>已初始化模型，等待检测')
        self.btn_i.clicked.connect(self.btn_i_func)
        self.btn_v.clicked.connect(self.btn_v_func)
        self.btn_sel.clicked.connect(self.btn_sel_func)
        self.btn.clicked.connect(self.btn_func)
        self.thread_i=thread_img()
        self.thread_i.signal.connect(self.img_signal)
        self.thread_v=thread_video()
        self.thread_v.signal1.connect(self.video_signal1)
        self.thread_v.signal2.connect(self.video_signal2)
        label_img=QPixmap('init0.jpg')
        # self.img_label.setScaledContents(True)
        self.img_label.setPixmap(label_img)

    def btn_func(self):
        if not mode:
            self.thread_i.start()
        else:
            self.thread_v.start()
            self.btn.setDisabled(True)
            self.print.append('>开启本地摄像头:0   (Esc键退出)')



    def img_signal(self,path):
        t1=time()
        r=detect(path)
        t2=time()
        t=t2-t1-0.15
        obj_num=len(r)
        self.print.clear()

        if not obj_num:
            self.print.append('>检测到0个目标')
            self.print.append('>用时:     {:.3f}s'.format(t))
        else:
            self.print.append('>检测到{}个目标'.format(obj_num))
            img_read(temp_path)
            img = QPixmap(temp_path)
            self.img_label.setPixmap(img)
            for index,i in enumerate(r):
                self.print.append(">目标{}:   {}     置信度:{:.2f}".format(index,i[0],i[1]))
                if i[0]=="无口罩":
                    Beep(3000, 80)
                    Beep(3000, 80)
            self.print.append('>用时:     {:.3f}s'.format(t))


    def video_signal1(self,e):
        print(e)
        self.btn.setEnabled(True)
        self.print.append('>已退出视频检测')
    def video_signal2(self,result):
        print(result)

    def btn_i_func(self):
        global mode
        self.print.clear()
        self.print.append('>已选择图片检测模式,请选择文件')
        mode=0
    def btn_v_func(self):
        global  mode
        mode=1
        self.print.clear()
        self.print.append('>已选择视频检测模式')
        label_img=QPixmap('init0.jpg')
        # self.img_label.setScaledContents(True)
        self.img_label.setPixmap(label_img)
    def btn_sel_func(self):
        global path
        self.btn.setDisabled(True)
        path,r=QFileDialog.getOpenFileName(
            self,
            caption="选择要检测的图片",
            directory=r"../predict/res/detection/",
            filter="文件格式(*.jpg)"
        )
        if r:
            file_name=os.path.basename(path)
            self.print.append(">已选择:'{}'".format(file_name))
            self.comboBox.clear()
            self.comboBox.addItem(file_name)
            img_read(path)
            img=QPixmap(temp_path)
            self.img_label.setScaledContents(True)
            self.img_label.setPixmap(img)
            self.btn.setEnabled(True)
        else:
            self.print.append(">未选择文件")
            label_img = QPixmap('init1.png')
            self.img_label.setScaledContents(True)
            self.img_label.setPixmap(label_img)


class thread_img(QThread):
    signal=pyqtSignal(str)
    def __init__(self):
        super(thread_img, self).__init__()

    def run(self) -> None:
        self.signal.emit(path)

class thread_video(QThread):
    signal1=pyqtSignal(str)
    signal2=pyqtSignal(list)
    def __init__(self):
        super(thread_video, self).__init__()
    def run(self) -> None:
        yolo = YOLO()
        cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

        while True:
            r, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame,result=yolo.detecter(frame)
            frame=np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.signal2.emit(result)
            cv2.imshow("mask detection", frame)
            c = cv2.waitKey(1)
            if c == 27:
                cap.release()
                cv2.destroyAllWindows()
                e='结束线程'
                break
        self.signal1.emit(e)










if __name__ =='__main__':
    app=QApplication(sys.argv)
    win=window()
    win.show()
    sys.exit(app.exec_())



