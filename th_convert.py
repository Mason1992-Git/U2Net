import os
import cv2
import numpy as np

img_path = r"E:\YL_Project\Projects\XiAn\U2NET_Split\09-UNET-EyeballSegmentation\TRAIN_DATA_JYZ\image"
mask_path = r"E:\YL_Project\Projects\XiAn\U2NET_Split\09-UNET-EyeballSegmentation\TRAIN_DATA_JYZ\mask"
save_path = r"E:\YL_Project\Projects\XiAn\U2NET_Split\09-UNET-EyeballSegmentation\TRAIN_DATA_JYZ\label"

# 将mask转为label
for img_name in os.listdir(mask_path):
    img_path = os.path.join(mask_path,img_name)
    # print(img_path)
    img_src = cv2.imread(img_path, 0)
    ret, mask = cv2.threshold(img_src, 10, 255, cv2.THRESH_BINARY)
    img_name = img_name[:-6]+".png"
    cv2.imwrite(os.path.join(save_path, img_name), mask)


# # 将多标签mask转换为label
# for img_name in os.listdir(img_path):
#     mask1 = img_name[:-4]+"_1.png"
#     mask2 = img_name[:-4] + "_2.png"
#     print(mask1)
#     img1 = cv2.imread(os.path.join(mask_path,mask1),0)
#     # img11 = np.copy(img1)
#     img2 = cv2.imread(os.path.join(mask_path,mask2),0)
#     h,w= img1.shape
#     print(h,w)
#     label = np.zeros((h,w),dtype=np.uint8)
#     print(label.shape)
#     for row in range(h):
#         for col in range(w):
#             if(img1[row,col] == 128 and img2[row,col] == 0):
#                 label[row,col] = 128
#             elif(img1[row,col] == 0 and img2[row,col] == 255):
#                 label[row,col] = 255
#             elif(img1[row,col] == 128 and img2[row,col] == 255):
#                 label[row,col] = 128
#             # else:
#             #     print(0)
#             # pass
#     label_name = img_name[:-4] + ".png"
#     cv2.imwrite(os.path.join(save_path,label_name),label)



# 创建黑色背景
# for img_name in os.listdir(path):
#     img_path = os.path.join(path, img_name)
#
#     img_src = cv2.imread(img_path, 0)
#     # print(img_src.shape)
#     h ,w= img_src.shape[0],img_src.shape[1]
#     balck_img = np.zeros([h,w],dtype=np.uint8)
#     img_name = img_name[:-3] + "png"
#     print(img_name)
#     cv2.imwrite(os.path.join(save_path, img_name), balck_img)


