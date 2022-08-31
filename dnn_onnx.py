import cv2
import os
from PIL import Image
import torch
import torchvision

if __name__ == '__main__':
    img_path = r"D:\DXXJ_TEST\lmqs"
    weigth_path = "onnx_saved/XJLM.onnx"
    save_path = r"D:\DXXJ_TEST\onnx_results"
    try:
        net = cv2.dnn.readNetFromONNX(weigth_path)
        print("load model successful!")
    except:
        print("Load Model Error...")
        exit(1)
    for img_name in os.listdir(img_path):
        img = cv2.imread(os.path.join(img_path,img_name))
        blob = cv2.dnn.blobFromImage(img,scalefactor= 1.0/255,size=(256,256),mean=(0,0,0),swapRB=True,crop=False)
        print(blob.shape)

        net.setInput(blob)

        outs = net.forward()
        # print(outs)
        # print(outs[0])
        # print(outs.shape)

        out = torch.tensor(outs)
        # print(out)
        out = out.cpu().clone().squeeze(0)
        img1 = torchvision.transforms.ToPILImage()(out)
        # print(out.shape)
        # im = Image.fromarray(outs)
        img1.save(os.path.join(save_path,img_name))
        #
        # os.system("pause")

