from model import U2NET
from model import U2NETP
import os
import torch
from PIL import ImageFont,Image,ImageDraw
import torchvision
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import torch.onnx
from torch import jit


if __name__ == '__main__':

    weight_path = r"weight\250.pth"
    # img_save_path = r"D:\DRIVE\test\img_save_path"
    onnx_save_path = r"onnx_saved\ZXMJ.onnx"

    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    model_name = "u2netp"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    if (model_name == 'u2net'):
        net = U2NET(3, 1).to(device)
    elif (model_name == 'u2netp'):
        net = U2NETP(3, 1).to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        net.eval()
        print("load weight successfully!")
    batch_size = 1
    inputs = torch.randn(1,3,256,256)
    torch_model = jit.trace(net,inputs.to(device))
    torch_model.save("onnx_saved\XZXMJ.pt")
    # 导出onnx
    u2net_input = torch.randn(batch_size,3,256,256)
    torch.onnx.export(net,u2net_input.to(device),onnx_save_path,input_names=["input"],output_names=["output"],opset_version=11,dynamic_axes={"input":{0:"batch_size"},"output":{0:"batch_size"}})
