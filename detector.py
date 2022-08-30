from UNet import MainNet
import os
import torch
from PIL import ImageFont,Image,ImageDraw
import torchvision
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
def compare(face1, face2):
    face1_norm = F.normalize(face1)
    face2_norm = F.normalize(face2)
    # print(face1_norm.shape)
    # print(face2_norm.shape)
    cosa = torch.matmul(face1_norm, face2_norm.t())
    return cosa
def compare_img(img1,img2):
    img1 = torch.tensor(np.array(img1), dtype=torch.float32)
    img2 = torch.tensor(img2.detach().numpy(),dtype=torch.float32)
    img1 = img1.reshape(1, -1)
    img2 = img2.reshape(1, -1)
    cos = compare(img1, img2)
    return cos
if __name__ == '__main__':
    img_path = r"d:\DRIVE\test\images"
    weight_path = r"weight\net.pth"
    img_save_path = r"D:\DRIVE\test\img_save_path"
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    net = MainNet().cuda()
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("load weight successfully!")

    for i,_name in enumerate(os.listdir(img_path)):
        _img =Image.open(os.path.join(img_path,_name))
        # print(_img.shape)
        w,h = _img.size
        # print(w,h)
        black = torchvision.transforms.ToPILImage()(torch.zeros(3, 256, 256))

        max_size = max(w,h)
        ratio = 256/max_size
        img = _img.resize((int(w*ratio),int(h*ratio)))

        black.paste(img,(0,0,int(w*ratio),int(h*ratio)))
        # black.show()
        input = torch.unsqueeze(transform(black),dim=0)
        # print(input.shape)
        input = input.cuda()
        out = net(input)

        x = out.cpu().clone().squeeze(0)
        # print(torch.max(x))
        # exit()
        # print(x.shape)
        # num = x.detach().numpy()
        img1 = torchvision.transforms.ToPILImage()(x)

        __name = _name[:2]

        name = __name + "_manual1-0000.jpg"
        # print(name)
        img2 = Image.open(os.path.join(r"D:\DRIVE\test\1st_mannual",name))
        w2, h2 = img2.size
        black2 = torchvision.transforms.ToPILImage()(torch.zeros(3, 256, 256))

        max_size2 = max(w2, h2)
        ratio2 = 256 / max_size2
        img2 = img2.resize((int(w2 * ratio2), int(h2 * ratio2)))

        black2.paste(img2, (0, 0, int(w * ratio), int(h * ratio)))

        x_o = transform(black)
        x_p = out[0].cpu()
        y_o =transform(black2)
        # print(y.shape)
        _img_show = torch.stack([x_o, x_p, y_o], 0)
        # print(_img_show.shape)
        # img_show = torchvision.transforms.ToPILImage()(_img_show)
        save_image(_img_show, os.path.join(img_save_path, f'{_name}.png'))
        cos = compare_img(y_o,x_p)
        print("文件名>>>:",_name)
        print("相似度为>>>:",cos)
        print("saved successfully !")
        # img_show.show()

