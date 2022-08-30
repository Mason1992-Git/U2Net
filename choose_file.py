import os
from shutil import  copyfile
path = r"E:\YL_Project\Projects\XiAn\U2NET_Split\09-UNET-EyeballSegmentation\分类\0"
src_path = r"E:\YL_Project\Projects\XiAn\U2NET_Split\09-UNET-EyeballSegmentation\cotter\cotter"
save_path = r"E:\YL_Project\Projects\XiAn\U2NET_Split\09-UNET-EyeballSegmentation\分类\0-0"

if __name__ == '__main__':
    for img_name in os.listdir(path):

        img_name = img_name[:-4]
        # print(img_name)
        try:
            copyfile(os.path.join(src_path,img_name),os.path.join(save_path,img_name))
        except:
            print("no this file...")