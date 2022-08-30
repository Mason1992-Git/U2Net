import os
import torch
from PIL import ImageFont,Image,ImageDraw
import torchvision
import torch.nn.functional as F
import numpy as np

def compare(face1, face2):
    face1_norm = F.normalize(face1)
    face2_norm = F.normalize(face2)
    # print(face1_norm.shape)
    # print(face2_norm.shape)
    cosa = torch.matmul(face1_norm, face2_norm.t())
    return cosa
def compare_img(img1,img2):
    img1 = torch.tensor(np.array(img1), dtype=torch.float32)
    img2 = torch.tensor(np.array(img2),dtype=torch.float32)
    img1 = img1.reshape(1, -1)
    img2 = img2.reshape(1, -1)
    cos = compare(img1, img2)
    return cos

if __name__ == '__main__':
    img1 = torch.tensor(np.array(Image.open("img/36_manual1-0000.jpg")),dtype=torch.float32)
    img2 = torch.tensor(np.array(Image.open("img/36_manual1-0000.jpg")),dtype=torch.float32)
    # print(img1.shape)
    img1 = img1.reshape(1,-1)
    # print(img1.shape)

    img2 = img2.reshape(1,-1)

    cos = compare(img1,img2)
    print(cos)