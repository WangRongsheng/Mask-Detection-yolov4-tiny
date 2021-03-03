from yolo_net.CSPDarknet53 import CSPDarknet53_tiny
from yolo_net.Yolo_Net import YoloBlock
import torch
from torchviz import make_dot
from os import getcwd
wd=getcwd()
input=torch.rand(1,3,416,416)
# net=CSPDarknet53_tiny(None)
net=YoloBlock(5,2)
out=net(input)
g=make_dot(out,params=dict(net.named_parameters()))
g.render(filename=wd+'/log/YoloBlock_h',view=False,format='pdf')
# g.view()