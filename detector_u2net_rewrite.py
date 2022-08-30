
from model import U2NET
from model import U2NETP
import os
import torch
from PIL import ImageFont,Image,ImageDraw
import torchvision
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import cv2

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

def FillHole(im_in):
    '''
    图像说明：
    图像为二值化图像，255白色为目标物，0黑色为背景
    要填充白色目标物中的黑色空洞
    '''
    # im_in = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE);

    # 复制 im_in 图像
    im_floodfill = im_in.copy()

    # Mask 用于 floodFill，官方要求长宽+2
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # floodFill函数中的seedPoint必须是背景
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if (im_floodfill[i][j] == 0):
                seedPoint = (i, j)
                isbreak = True
                break
        if (isbreak):
            break
    # 得到im_floodfill
    cv2.floodFill(im_floodfill, mask, seedPoint, 255);

    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = im_in | im_floodfill_inv

    # 保存结果
    # cv2.imwrite(os.path.join(SavePath,img_name), im_out)
    return im_out
if __name__ == '__main__':
    img_path = r"D:\dyx_test"
    weight_path = r"weight\100.pth"
    img_save_path = r"test_save"
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    model_name = "u2net"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    if (model_name == 'u2net'):
        net = U2NET(3, 1).to(device)
    elif (model_name == 'u2netp'):
        net = U2NETP(3, 1).to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("load weight successfully!")
    # black = torchvision.transforms.ToPILImage()(torch.ones(3, 256, 256))
    # input = torch.unsqueeze(transform(black), dim=0)
    # input = input.to(device)
    # out = net(input)
    # x = out[0].cpu().clone().squeeze(0)
    # print(x)
    for i, cls0 in enumerate(os.listdir(img_path)):
        for cls1 in os.listdir(os.path.join(img_path,cls0)):
            for _name in os.listdir(os.path.join(img_path,cls0,cls1)):
                if _name.endswith("jpg") or _name.endswith("png"):
                    if not os.path.exists(os.path.join(img_path,cls0,cls1+"_mask0")):
                        os.makedirs(os.path.join(img_path,cls0,cls1+"_mask0"))
                    img_save_path = os.path.join(img_path,cls0,cls1+"_mask0")
                    print("name = ",_name)
                    _img =Image.open(os.path.join(img_path,cls0,cls1,_name))
                    f_size = torch.Tensor(_img.size).long()
                    print("f_size = ",f_size)
                    # 等比缩放
                    img1_size = torch.Tensor(_img.size)
                    l_max_index = img1_size.argmax()
                    ratio = 256 / img1_size[l_max_index.item()]
                    img1_re2size = img1_size * ratio
                    img1_re2size = img1_re2size.long()
                    img = _img.resize(img1_re2size)
                    w, h = img1_re2size.tolist()
                    black1 = torchvision.transforms.ToPILImage()(torch.zeros(3, 256, 256))
                    black1.paste(img, (0, 0, int(w), int(h)))
                    # img = _img.resize((256,256))
                    # img = torchvision.transforms.ToPILImage()(torch.ones(3, 256, 256))
                    input = torch.unsqueeze(transform(black1), dim=0)
                    input = input.to(device)
                    out = net(input)
                    x = out[0].cpu().clone().squeeze(0)
                    # print("x_shape>>",x.shape)
                    # print(x[0][100][255])
                    img1 = torchvision.transforms.ToPILImage()(x)
                    # img1 = img1.resize(f_size)
                    # print("img1.size =" ,img1.size)
                    # img1.show()
                    # save_image(img1, os.path.join(img_save_path, f'{_name}.png'))
                    img1.save(os.path.join(img_save_path, f'{_name}.png'))

                    # 等比缩放
                    if not os.path.exists(os.path.join(img_path,cls0,cls1+"_src")):
                        os.makedirs(os.path.join(img_path,cls0,cls1+"_src"))
                    # _img = Image.open(os.path.join(path, img_name))
                    img_save_path_src = os.path.join(img_path,cls0,cls1+"_src")
                    f_size = torch.Tensor(_img.size).long()
                    # print("f_size = ", f_size)
                    # 等比缩放
                    img1_size = torch.Tensor(_img.size)
                    l_max_index = img1_size.argmax()
                    ratio = 256 / img1_size[l_max_index.item()]
                    img1_re2size = img1_size * ratio
                    img1_re2size = img1_re2size.long()
                    img = _img.resize(img1_re2size)
                    w, h = img1_re2size.tolist()
                    black1 = torchvision.transforms.ToPILImage()(torch.zeros(3, 256, 256))
                    black1.paste(img, (0, 0, int(w), int(h)))
                    black1.save(os.path.join(img_save_path_src, _name))


                    # crop
                    path_src = os.path.join(img_save_path_src, _name)
                    path_mask = os.path.join(img_save_path,_name+".png")
                    try:
                        img_src = cv2.imread(path_src, 0)
                        #单通道图像
                        print("img_src = ", img_src.shape)
                    except:
                        continue
                    img_mask = cv2.imread(path_mask, 0)
                    # print("img_mask = ", img_mask.shape)
                    ret, mask = cv2.threshold(img_mask, 160, 255, cv2.THRESH_BINARY)
                    mask = FillHole(mask)
                    img1_bg = cv2.bitwise_and(img_src, img_src, mask=mask)
                    # cv2.imwrite(os.path.join(mask_path,img_name),mask)
                    if not os.path.exists(os.path.join(img_path,cls0,cls1+"_crop")):
                        os.makedirs(os.path.join(img_path,cls0,cls1+"_crop"))
                    crop_apth = os.path.join(img_path,cls0,cls1+"_crop")
                    cv2.imwrite(os.path.join(crop_apth, _name), img1_bg)






