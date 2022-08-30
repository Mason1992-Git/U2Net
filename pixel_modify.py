import cv2
import os
import numpy as np
import shutil
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

src_path = r"D:\dyx_test\src\1_src"
path = r"D:\dyx_test\src\1_mask0"
save_path = r"C:\Users\1\Desktop\dyx_defect\mask"
defect_save_path = r"C:\Users\1\Desktop\dyx_defect\defect"
# save_path = r"D:\dropper_test\defect\defect\3"
for img_name in os.listdir(path):
    print(os.path.join(path,img_name))
    img_mask = cv2.imread(os.path.join(path,img_name), 0)
    ret, thresh = cv2.threshold(img_mask, 160, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    mask = FillHole(result)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    print("num_labels = ",num_labels)
    print("stats = ",stats)
    areas = []
    rec_length = []
    for i in range(num_labels):
        areas.append(stats[i][-1])
        rec_length.append(max(stats[i][2],stats[i][3]))
        # print("轮廓%d的面积:%d" % (i, stats[i][-1]))
    # print(areas[:])
    # print(areas[1:])
    # print("len = ",len(areas[1:]))
    # print(rec_length)
    if len(areas[1:]) == 0:
        print("defect_name = ", img_name[:-4])
        shutil.copy2(os.path.join(src_path, img_name[:-4]), os.path.join(defect_save_path, img_name[:-4]))
        # 保存开运算结果
        cv2.imwrite(os.path.join(save_path, img_name), result)
        continue
    area_max = np.max(areas[1:])
    total_area = np.sum(areas[1:])
    rec_len_max = np.max(rec_length[1:])
    print("max_aera = ",area_max)
    print("total_area = ", total_area)
    print("rec_len_max = ", rec_len_max)
    if total_area > 1000:
        continue
    if area_max < 50 and rec_len_max < 50:
        print("defect_name = ", img_name[:-4])
        shutil.copy2(os.path.join(src_path, img_name[:-4]), os.path.join(defect_save_path, img_name[:-4]))
        # 保存开运算结果
        cv2.imwrite(os.path.join(save_path, img_name), result)
    # if (area_max < 50 or total_area < 300) or  rec_len_max < 60:
    #     print("defect_name = ",img_name[:-4])
    #     shutil.copy2(os.path.join(src_path,img_name[:-4]),os.path.join(defect_save_path,img_name[:-4]))





    # count = 0
    # for i in range(256):
    #     for j in range(256):
    #         count += 1
    #         value = img_mask[i,j]
    #         if abs(value-100)<10:
    #             img_mask[i, j] = 100
    #         elif abs(value-255)<10:
    #             img_mask[i, j] = 255
    #         else:
    #             img_mask[i, j] = 0
    #         # print(value)
    # cv2.imwrite(os.path.join(save_path,img_name),img_mask)
    # print("count = ",count)
    # pixel_value = img_mask.item(10,10,2)
    # print(pixel_value)

