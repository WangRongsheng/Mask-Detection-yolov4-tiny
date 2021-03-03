from PIL import Image
from yolo_net.Class_Yolo import YOLO

yolo=YOLO()
def compress(img,rate):
    iw,ih=img.size
    if min(iw,ih)>800:
        nw=int(rate*iw)
        nh=int(rate*ih)
        nimg=img.resize((nw,nh))
        print('原图压缩至:{}'.format(nimg.size))
        return nimg
    else:
        print('无需压缩')
        return img

def detect(path):
    image=Image.open(path)
    image=compress(image,0.3)
    out_img,result=yolo.detecter(image)
    out_img.save(r'D:\PycharmProjects\Mask Detection\predict\res\cache\temp.jpg')
    return result
