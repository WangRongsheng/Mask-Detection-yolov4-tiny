import xml.etree.ElementTree as et
import numpy as np
import glob

def iou(box,cluster):
    x=np.minimum(cluster[:,0],box[0])
    y=np.minimum(cluster[:,1],box[1])
    i=x*y
    area1=box[0]*box[1]
    arer2=cluster[:,0]*cluster[:,1]
    IoU=i/(area1+arer2-i)
    return IoU
def kmeans(box,k):
    box_num=box.shape[0]
    distance=np.empty((box_num,k))
    clu=np.zeros((box_num,))
    np.random.seed()
    cluster=box[np.random.choice(box_num,k,replace=False)]
    while True:
        for i in range(box_num):
            distance[i]=1-iou(box[i],cluster)
        nearst=np.argmin(distance,axis=1)
        if (clu==nearst).all():
            break
        for j in range(k):
            cluster[j]=np.median(box[nearst==j],axis=0)
        clu=nearst
    return cluster
def read_xml(xmlpath):
    data=[]
    for xml in glob.glob('{}\*xml'.format(xmlpath)):
        tree = et.parse(xml)
        root = tree.getroot()
        for i in root.iter('size'):
            h = int(i.find('height').text)
            w = int(i.find('width').text)
        for i in root.iter('object'):
            box = i.find('bndbox')
            xmin = int(box.find('xmin').text) / w
            xmax = int(box.find('xmax').text) / w
            ymin = int(box.find('ymin').text) / h
            ymax = int(box.find('ymax').text) / h
            xmin = np.float64(xmin)
            xmax = np.float64(xmax)
            ymin = np.float64(ymin)
            ymax = np.float64(ymax)
            data.append([xmax-xmin,ymax-ymin])
    return np.array(data)

if __name__=='__main__':
    xmlpath=r'../dataset/Annotations'
    size=416
    anc_num=6
    hwdata=read_xml(xmlpath)
    out_anchor=kmeans(hwdata,anc_num)
    out_anchor=out_anchor[np.argsort(out_anchor[:,0])]
    hwdata=out_anchor*416
    f=open('kmeans_anchor.txt', 'w')
    num=np.shape(hwdata)[0]
    for i in range(num):
        if i==0:
            x_y="%d,%d"%(hwdata[i][0],hwdata[i][1])
        else:
            x_y=", %d,%d"%(hwdata[i][0],hwdata[i][1])
        f.write(x_y)
    f.close()
    print('已生成anchor')