
import numpy as np
from PIL import Image
from yolo_net.Class_Yolo import YOLO
import  cv2


yolo=YOLO()

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
while True:
    r,frame=cap.read()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame=Image.fromarray(np.uint8(frame))
    frame=np.array(yolo.detecter(frame)[0])
    frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    cv2.imshow("detector",frame)
    c=cv2.waitKey(1)
    if c==27:
        cap.release()
        cv2.destroyAllWindows()
        break
