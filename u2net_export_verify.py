import cv2
import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import ImageFont,Image,ImageDraw
import torchvision
import shutil
import onnxruntime
import math
from onnxruntime.datasets import get_example

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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
if __name__ == '__main__':
    weight_path = r"XJTC_160GPU.pt"
    onnx_path = "onnx_saved\ZXMJ.onnx"
    img_path = r"C:\Users\1\Documents\Tencent Files\1139417291\FileRecv\contact_line_anchor\contact_line_anchor"
    save_path = r"D:\DXXJ_TEST\onnx_results"
    defect_save_path = r"D:\DXXJ_TEST\DEFECT_RESULT"
    device = torch.device("cuda")
    NETWORK_WIDTH = 256
    SCALE_WIDTH = 0
    # 使用onnx或者torchscript
    torchscript_flag = False
    onnx_flag = t = True
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if torchscript_flag:
        if os.path.exists(weight_path):
            model = torch.jit.load(weight_path)
            print("load torchscript weight successfully!")
            model.eval()
    if onnx_flag:
        model = onnxruntime.InferenceSession(onnx_path)
        print("load onnx weight successfully!")
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        print('Input Name:', input_name)
        print('Output Name:', output_name)
    #     example_model = get_example(r"G:\09-UNET-EyeballSegmentation\onnx_saved\XJLM.onnx")
    #     sess = onnxruntime.InferenceSession(r"G:\09-UNET-EyeballSegmentation\onnx_saved\XJLM.onnx",None)
    # # dummy_input = torch.zeros(1, 3, 256, 256, device='cuda')
    # ort_inputs = {sess.get_inputs()[0].name: to_numpy(dummy_input)}
    # out = sess.run([sess.get_outputs()[0].name], ort_inputs)
    # out = torch.tensor(out)
    # x1 = out[0].cpu().clone().squeeze(0)
    # print("onnx_out:", x1)
    # img1 = torchvision.transforms.ToPILImage()(x1)
    # print("onnx_out:", img1)
    # # img1.show()
    # img1.save(os.path.join(save_path,'onnx.png'))
    #
    # pt_out = model(dummy_input)
    # x2 = pt_out[0].cpu().clone().squeeze(0)
    # print("pt_out:", x2)
    # img2 = torchvision.transforms.ToPILImage()(x2)
    # print("pt_out:",img2)
    # img2.save(os.path.join(save_path, 'pt.png'))



    for _name in os.listdir(os.path.join(img_path)):
        if _name.endswith("jpg") or _name.endswith("png"):
            print("开始分割...")
            print(os.path.join(img_path,_name))
            if not os.path.exists(os.path.join(save_path,"_mask0")):
                os.makedirs(os.path.join(save_path,"_mask0"))
            img_save_path = os.path.join(save_path,"_mask0")
            # print("name = ",_name)
            _img = Image.open(os.path.join(img_path,_name))
            f_size = torch.Tensor(_img.size).long()
            # print("f_size = ",f_size)
            # 等比缩放
            img1_size = torch.Tensor(_img.size)
            l_max_index = img1_size.argmax()
            ratio = NETWORK_WIDTH / img1_size[l_max_index.item()]
            img1_re2size = img1_size * ratio
            img1_re2size = img1_re2size.long()
            img = _img.resize(img1_re2size)
            w, h = img1_re2size.tolist()
            black1 = torchvision.transforms.ToPILImage()(torch.zeros(3, NETWORK_WIDTH, NETWORK_WIDTH))
            black1.paste(img, (0, 0, int(w), int(h)))
            # img = _img.resize((256,256))
            # img = torchvision.transforms.ToPILImage()(torch.ones(3, 256, 256))
            input = torch.unsqueeze(transform(black1), dim=0)
            input = input.to(device)
            # print("input_shape>>:", input.shape)

            if torchscript_flag:
                out = model(input)
            if onnx_flag:
                ort_inputs = {model.get_inputs()[0].name: to_numpy(input)}
                out = model.run(None, ort_inputs)
                # print("type_out = ",type(out))
                # print(len(out))
                out = torch.tensor(out)
            # print("out_shape>>",out.size())
            x = out[0].cpu().clone().squeeze(0)
            # print("x_shape>>", x.shape)
            img1 = torchvision.transforms.ToPILImage()(x)
            img1.save(os.path.join(img_save_path, f'{_name}.png'))


            # 等比缩放
            if not os.path.exists(os.path.join(save_path, "_src")):
                os.makedirs(os.path.join(save_path,"_src"))
            # _img = Image.open(os.path.join(path, img_name))
            img_save_path_src = os.path.join(save_path,"_src")
            f_size = torch.Tensor(_img.size).long()
            # print("f_size = ", f_size)
            # 等比缩放
            img1_size = torch.Tensor(_img.size)
            l_max_index = img1_size.argmax()
            ratio = NETWORK_WIDTH / img1_size[l_max_index.item()]
            img1_re2size = img1_size * ratio
            img1_re2size = img1_re2size.long()
            img = _img.resize(img1_re2size)
            w, h = img1_re2size.tolist()
            SCALE_WIDTH = w
            black1 = torchvision.transforms.ToPILImage()(torch.zeros(3, NETWORK_WIDTH, NETWORK_WIDTH))
            black1.paste(img, (0, 0, int(w), int(h)))
            black1.save(os.path.join(img_save_path_src, _name))

            # crop
            path_src = os.path.join(img_save_path_src, _name)
            path_mask = os.path.join(img_save_path, _name + ".png")
            try:
                img_src = cv2.imread(path_src, 0)
                # 单通道图像
                # print("img_src = ", img_src.shape)
            except:
                continue
            img_mask = cv2.imread(path_mask, 0)
            # print("img_mask = ", img_mask.shape)
            ret, mask = cv2.threshold(img_mask, 1, 255, cv2.THRESH_BINARY)
            if not os.path.exists(os.path.join(save_path,"_mask1")):
                os.makedirs(os.path.join(save_path,"_mask1"))
            imgMASK1_save_path = os.path.join(save_path,"_mask1")
            cv2.imwrite(os.path.join(imgMASK1_save_path, _name), mask)
            # mask = FillHole(mask)
            img = np.ones((NETWORK_WIDTH, NETWORK_WIDTH), dtype=np.uint8)
            bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            bgr_img[:, :, 0] = 0
            bgr_img[:, :, 1] = 0
            bgr_img[:, :, 2] = 0
            b, g, r = cv2.split(bgr_img)
            for i in range(NETWORK_WIDTH):
                for j in range(NETWORK_WIDTH):
                    pixel = img_mask[i][j]
                    if pixel > 220:
                        r[i][j] = 255
                    # elif pixel > 60:
                    #     g[i][j] = 255
                    elif pixel > 1:
                        g[i][j] = 255
                    else:
                        b[i][j] = 0
            merged = cv2.merge([b, g, r])
            # cv2.imshow("merged", merged)
            # img_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            img_src_ = cv2.imread(os.path.join(img_save_path_src, _name),1)
            dst = cv2.addWeighted(img_src_, 1, merged, 0.2, 0)
            # cv2.imshow("crop",dst)
            # cv2.waitKey(0)
            # img1_bg = cv2.bitwise_and(img_src, img_src, mask=mask)
            # cv2.imwrite(os.path.join(mask_path,img_name),mask)
            if not os.path.exists(os.path.join(save_path,"_crop")):
                os.makedirs(os.path.join(save_path, "_crop"))
            crop_apth = os.path.join(save_path,"_crop")
            cv2.imwrite(os.path.join(crop_apth, _name), dst)
    print("分割完成...")
    # save_path = r"D:\dropper_test\defect\defect\3"
    print("开始判定缺陷....")
    for img_name in os.listdir(os.path.join(save_path,"_mask0")):
        print(os.path.join(os.path.join(save_path,"_mask0"), img_name))
        img_mask = cv2.imread(os.path.join(os.path.join(save_path,"_mask0"), img_name), 0)
        ret, thresh = cv2.threshold(img_mask, 200, 255, cv2.THRESH_BINARY)
        x,y = thresh.shape
        kernel = np.ones((3, 3), np.uint8)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_dilate= cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # #开运算
        # result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        #腐蚀

        dst = cv2.dilate(thresh, kernel_dilate)
        #膨胀
        result = cv2.erode(dst, kernel_erode)
        # cv2.imshow("open",result)
        # cv2.waitKey(0)
        # # 保存开运算结果
        if not os.path.exists(os.path.join(save_path, "_open")):
            os.makedirs(os.path.join(save_path, "_open"))
        open_path = os.path.join(save_path, "_open")
        cv2.imwrite(os.path.join(open_path, img_name), thresh)
        # mask = FillHole(result)
        #####################################################
        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # # print("num_labels = ", num_labels)
        # # print("stats = ", stats)
        # areas = []
        # rec_length = []
        # for i in range(num_labels):
        #     areas.append(stats[i][-1])
        #     rec_length.append(max(stats[i][2], stats[i][3]))
        #     # print("轮廓%d的面积:%d" % (i, stats[i][-1]))
        # # print(areas[:])
        # # print(areas[1:])
        # # print("len = ",len(areas[1:]))
        # # print(rec_length)
        # if len(areas[1:]) == 0:
        #     print("defect_name = ", img_name[:-4])
        #     shutil.copy2(os.path.join(os.path.join(save_path,"_src"), img_name[:-4]), os.path.join(defect_save_path, img_name[:-4]))
        #     continue
        # total_area = np.sum(areas[1:])
        # area_max = np.max(areas[1:])
        # rec_len_max = np.max(rec_length[1:])
        # # print("max_aera = ", area_max)
        # print("total_area = ", total_area)
        # # print("rec_len_max = ", rec_len_max)
        # if total_area <100:
        #     print("defect_name = ", img_name[:-4])
        #     shutil.copy2(os.path.join(os.path.join(save_path,"_src"), img_name[:-4]), os.path.join(defect_save_path, img_name[:-4]))
        ###################################################################

        contours, hierarchy = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        scale_list = []
        max_aera = 0
        all_aera = 0
        max_rect_length = 0
        all_length = 0
        min_angle = 1000
        for i in range(len(contours)):
            # 第i个轮廓的面积.
            # aera1 = abs(cv2.contourArea(contours[i], True))
            # print("连通域面积：", abs(aera1))
            single_masks = np.zeros((x, y))
            fill_image = cv2.fillConvexPoly(single_masks, contours[i], 255)
            pixels = cv2.countNonZero(fill_image)
            all_aera += pixels
            if pixels > max_aera:
                max_aera = pixels
            # print("连通域面积2：", abs(pixels))
            # 第i个轮廓的最小外接矩形
            min_rect = cv2.minAreaRect(contours[i])
            w_rect = min_rect[1][0]
            h_rect = min_rect[1][1]
            temp_angle = min_rect[2]
            if temp_angle < min_angle:
                min_angle = temp_angle
            temp = max(w_rect, h_rect)
            all_length += temp
            if temp > max_rect_length :
                max_rect_length = temp
            aera_rect = w_rect * h_rect
            try:
                scale = pixels / aera_rect
            except:
                scale = 1

            # print("rect_aera:", aera_rect)
            # print("scale = ", scale)
            scale_list.append([scale,pixels])
        # 最小比例大于0.4，证明有曲线
        scale_list.sort()
        # print("scale_list = ", scale_list)
        # 保存缺陷结果
        if not os.path.exists(os.path.join(defect_save_path, "_src")):
            os.makedirs(os.path.join(defect_save_path, "_src"))
        defect_save_src_path = os.path.join(defect_save_path, "_src")
        if not os.path.exists(os.path.join(defect_save_path, "mask")):
            os.makedirs(os.path.join(defect_save_path, "mask"))
        defect_save_mask_path = os.path.join(defect_save_path, "mask")
        print("angle = ", min_angle)
        print("max_zera = ",max_aera)
        print("all_aera = ", all_aera)
        print("rect_length = ",max_rect_length)
        if min_angle > 45:
            contact_length = SCALE_WIDTH / math.sin(min_angle*math.pi/180)
        else:
            contact_length = SCALE_WIDTH / math.sin((90- min_angle) * math.pi / 180)
        print("***********")
        print("sin_angle = ",math.sin(min_angle*math.pi/180))
        print("SCALE_WIDTH = ",SCALE_WIDTH)
        print("all_length = ",all_length)
        print("contact_length = ",contact_length)
        print("=============================================")
        if max_rect_length > 150:
            if (contact_length -all_length) < 45:
                shutil.copy2(os.path.join(os.path.join(save_path, "_src"), img_name[:-4]),os.path.join(defect_save_src_path, img_name[:-4]))
                shutil.copy2(os.path.join(os.path.join(save_path, "_mask0"), img_name[:-4]+".png"),os.path.join(defect_save_mask_path, img_name[:-4]+".png"))
        ############################################等压线缺陷判断
        # if all_aera < 150 and max_rect_length < 40:
        #     shutil.copy2(os.path.join(os.path.join(save_path, "_src"), img_name[:-4]),os.path.join(defect_save_src_path, img_name[:-4]))
        #     shutil.copy2(os.path.join(os.path.join(save_path, "_mask0"), img_name[:-4]+".png"),os.path.join(defect_save_mask_path, img_name[:-4]+".png"))
        ####################################载流环缺陷判断
        # if (len(scale_list) > 0):
        #     if (scale_list[0][0] > 0.8):
        #         shutil.copy2(os.path.join(os.path.join(save_path,"_src"), img_name[:-4]), os.path.join(defect_save_src_path, img_name[:-4]))
        #         shutil.copy2(os.path.join(os.path.join(save_path, "_mask0"), img_name[:-4]+".png"),os.path.join(defect_save_mask_path, img_name[:-4]+".png"))
        #     elif(scale_list[0][0] < 0.4):
        #         continue
        #     elif(scale_list[0][1] < 120):
        #         shutil.copy2(os.path.join(os.path.join(save_path, "_src"), img_name[:-4]),os.path.join(defect_save_src_path, img_name[:-4]))
        #         shutil.copy2(os.path.join(os.path.join(save_path, "_mask0"), img_name[:-4] + ".png"),os.path.join(defect_save_mask_path, img_name[:-4] + ".png"))
        # else:
        #     shutil.copy2(os.path.join(os.path.join(save_path,"_src"), img_name[:-4]), os.path.join(defect_save_src_path, img_name[:-4]))
        #     shutil.copy2(os.path.join(os.path.join(save_path, "_mask0"), img_name[:-4] + ".png"),os.path.join(defect_save_mask_path, img_name[:-4] + ".png"))



