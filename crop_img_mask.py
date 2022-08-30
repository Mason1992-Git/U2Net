import os
import cv2
import numpy as np
from PIL import ImageFont,Image,ImageDraw
import torchvision
import torch
# path_src = r"C:\Users\dell\Desktop\test_split\40_032918303_K409312_125_3_12.jpg"
# path_mask = r"C:\Users\dell\Desktop\test_split\40_032918303_K409312_125_3_12_1.png"

path = r"G:\jueyuanzi1\right"
# u2net结果路径
save_path = r"E:\YL_Project\Projects\XiAn\U2NET_Split\09-UNET-EyeballSegmentation\test_save"
img_save_path = r"E:\YL_Project\Projects\XiAn\U2NET_Split\09-UNET-EyeballSegmentation\test_trans"
crop_path = r"E:\YL_Project\Projects\XiAn\U2NET_Split\09-UNET-EyeballSegmentation\crop"
mask_path = r"E:\YL_Project\Projects\XiAn\U2NET_Split\09-UNET-EyeballSegmentation\mask_path"





def FillHole(im_in, SavePath,img_name):
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
    cv2.imwrite(os.path.join(SavePath,img_name), im_out)
    return im_out


if __name__ == '__main__':

    # for img_name in os.listdir(path):
    #     _img = Image.open(os.path.join(path, img_name))
    #     f_size = torch.Tensor(_img.size).long()
    #     print("f_size = ", f_size)
    #     # 等比缩放
    #     img1_size = torch.Tensor(_img.size)
    #     l_max_index = img1_size.argmax()
    #     ratio = 256 / img1_size[l_max_index.item()]
    #     img1_re2size = img1_size * ratio
    #     img1_re2size = img1_re2size.long()
    #     img = _img.resize(img1_re2size)
    #     w, h = img1_re2size.tolist()
    #     black1 = torchvision.transforms.ToPILImage()(torch.zeros(3, 256, 256))
    #     black1.paste(img, (0, 0, int(w), int(h)))
    #     black1.save(os.path.join(img_save_path, img_name))



    for img_name in os.listdir(img_save_path):
        if img_name.endswith("jpg") or img_name.endswith("png"):
            path_src = os.path.join(img_save_path,img_name)
            path_mask = os.path.join(save_path,img_name+".png")
            print(img_name)

            img_src=  cv2.imread(path_src,0)
            print("img_src = ",img_src.shape)
            img_mask = cv2.imread(path_mask,0)
            print("img_mask = ", img_mask.shape)
            ret, mask = cv2.threshold(img_mask,160,255,cv2.THRESH_BINARY)
            # kernel = np.ones((8, 8), dtype=np.uint8)
            # 腐蚀
            # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))
            # cv2.erode(mask,kernel1,1)
            # 膨胀
            # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(30,30))
            # mask = cv2.dilate(mask, kernel2, 1)  # 1:迭代次数，也就是执行几次膨胀操作
            # # 开操作
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            # mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
            mask = FillHole(mask, mask_path, img_name)
            # mask_inv = cv2.bitwise_not(mask)
            img1_bg = cv2.bitwise_and(img_src, img_src, mask=mask)
            # cv2.imwrite(os.path.join(mask_path,img_name),mask)
            cv2.imwrite(os.path.join(crop_path, img_name), img1_bg)
