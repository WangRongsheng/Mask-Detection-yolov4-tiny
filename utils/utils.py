import torch.nn as nn
from torchvision.ops import nms
import numpy as np
import torch
from PIL import Image


def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou



class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self, input):
        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors]

        prediction = input.view(batch_size, self.num_anchors,
                                self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output.data


def letterbox(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def NMS(prediction,class_num,conf_thres=0.8,nms_thres=0.2):
    #   转换坐标形式:(x,y,w,h)>(x1,y1,x2,y2)
    box_temp=prediction.new(prediction.shape)
    box_temp[:,:,0]=prediction[:,:,0]-prediction[:,:,2]/2
    box_temp[:,:,1]=prediction[:,:,1]-prediction[:,:,3]/2
    box_temp[:,:,2]=prediction[:,:,0]+prediction[:,:,2]/2
    box_temp[:,:,3]=prediction[:,:,1]+prediction[:,:,3]/2
    prediction[:,:,:4]=box_temp[:,:,:4]

    #   output 存放nms输出
    output = [None for _ in range(len(prediction))]
    #   prediction:(batch_size,2535,7)
    #   7:4+1+2
    #   pred:2535*7
    for index,pred in enumerate(prediction):
        #   cls_conf:最大类别得分 cls_pred:最大类别索引
        cls_conf,cls_pred=torch.max(pred[:,5:7],dim=1,keepdim=True)
        score=pred[:,4]*cls_conf[:,0]
        size_a=pred.shape[0]
        pred=pred[(score>conf_thres).squeeze()]
        size_b=pred.shape[0]
        # print("根据置信度得分删除{}个预测框，还剩{}个预测框".format(size_a-size_b,size_b))

        cls_conf=cls_conf[(score>conf_thres).squeeze()]
        cls_pred=cls_pred[(score>conf_thres).squeeze()]

        if not pred.size(0):
            continue
        #   detection:4+obj_conf+最大类得分+最大类
        detection=torch.cat((pred[:,:5],cls_conf.float(),cls_pred.float()),dim=1)
        # print(detection)
        # print("*****"*15)
        #unique_label包含所有类别
        unique_label=detection[:,-1].cpu().unique()
        #对两个类分别进行NMS算法
        for cls in unique_label:
            is_cls=detection[:,-1]==cls
            # print("当前检测类{}".format(cls))
            #当前类存入detection_class
            detection_class=detection[is_cls]
            # print(detection_class)

            boxes=detection_class[:,:4]             #预测框坐标
            score=detection_class[:,4]*detection_class[:,5]

            # keep保存NMS过滤后的预测框索引
            keep=nms(boxes=boxes,scores=score,iou_threshold=nms_thres)
            # print(keep)
            # print("*****" * 15)
            max_detection=detection_class[keep]
            # print("*****" * 15)
            # print(max_detection)
            output[index] = max_detection if output[index] is None else torch.cat(
                (output[index], max_detection))
    return output





def remove_gray(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape*np.min(input_shape/image_shape)
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)/input_shape
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)/input_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)
    return boxes
