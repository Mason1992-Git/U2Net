import cv2
import os
from PIL import Image
import torch
import torchvision
import numpy as np

if __name__ == '__main__':
    img_path = r"G:\09-UNET-EyeballSegmentation\train_data\image"
    weigth_path = "onnx_saved/ZXMJ.onnx"
    save_path = r"D:\DXXJ_TEST\onnx_results"
    NETWORK_WIDTH = 256


    try:
        net = cv2.dnn.readNetFromONNX(weigth_path)
        print("load model successful!")
    except:
        print("Load Model Error...")
        exit(1)
    for img_name in os.listdir(img_path):
        img = cv2.imread(os.path.join(img_path,img_name))
        blob = cv2.dnn.blobFromImage(img,scalefactor= 1.0/255,size=(256,256),mean=(0,0,0),swapRB=True,crop=False)
        # print(blob.shape)
        net.setInput(blob)
        outs = net.forward()

        out = torch.tensor(outs)
        out = out.cpu().clone().squeeze(0)
        if not os.path.exists(os.path.join(save_path, "_mask0")):
            os.makedirs(os.path.join(save_path, "_mask0"))
        mask_save_path = os.path.join(save_path, "_mask0")
        img1 = torchvision.transforms.ToPILImage()(out)
        img1.save(os.path.join(mask_save_path,img_name))
        # 等比缩放
        if not os.path.exists(os.path.join(save_path, "_src")):
            os.makedirs(os.path.join(save_path, "_src"))
        img_save_path_src = os.path.join(save_path, "_src")
        black_img = np.zeros((NETWORK_WIDTH, NETWORK_WIDTH), dtype=np.uint8)
        w_b, h_b = black_img.shape
        w,h,c = img.shape
        max_num = max(w, h)
        ratio = w_b / max_num
        img_data = cv2.resize(img, (int(h * ratio), int(w * ratio)))
        black = Image.fromarray(cv2.cvtColor(black_img, cv2.COLOR_GRAY2RGB))
        # img_pil = Image.fromarray(cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB))
        img_pil = Image.fromarray(img_data)
        black.paste(img_pil, (0, 0))
        data = cv2.cvtColor(np.asarray(black), cv2.COLOR_RGB2GRAY)
        cv2.imwrite(os.path.join(img_save_path_src, img_name), data)
    print("分割完成......")
    for mask_name in os.listdir(os.path.join(save_path, "_mask0")):
        img_mask = cv2.imread(os.path.join(save_path,"_mask0",mask_name),0)
        img = np.ones((NETWORK_WIDTH, NETWORK_WIDTH), dtype=np.uint8)
        bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        bgr_img[:, :, 0] = 0
        bgr_img[:, :, 1] = 0
        bgr_img[:, :, 2] = 0
        b, g, r = cv2.split(bgr_img)
        for i in range(NETWORK_WIDTH):
            for j in range(NETWORK_WIDTH):
                pixel = img_mask[i][j]
                if pixel > 200:
                    r[i][j] = 255
                # elif pixel > 60:
                #     g[i][j] = 255
                elif 10>pixel > 1:
                    g[i][j] = 255
                else:
                    b[i][j] = 0
        merged = cv2.merge([b, g, r])
        img_src_ = cv2.imread(os.path.join(save_path, "_src", mask_name), 1)
        dst = cv2.addWeighted(img_src_, 1, merged, 0.2, 0)
        if not os.path.exists(os.path.join(save_path, "_crop")):
            os.makedirs(os.path.join(save_path, "_crop"))
        crop_apth = os.path.join(save_path, "_crop")
        cv2.imwrite(os.path.join(crop_apth, mask_name), dst)

