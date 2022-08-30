import cv2
import math
import numpy as np
import os


if __name__=='__main__':
    path = r"E:\YL_Project\Projects\DCL-master\dataset_test\CLS\1"
    save_path = r"E:\YL_Project\Projects\DCL-master\dataset_test\CLS\1"
    for file in os.listdir(path):
        if file.endswith("png") or file.endswith("jpg"):
            file_path = os.path.join(path,file)
            print(file_path)
            src = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),
                               cv2.IMREAD_ANYCOLOR)
            # src = cv2.imread(file_path,0)
            dst = cv2.flip(src,1)

            save_file_path = os.path.join(save_path,"mirror_"+file)
            if file.endswith("png"):
                cv2.imencode('.png', dst)[1].tofile(save_file_path)
            else:
                cv2.imencode('.jpg', dst)[1].tofile(save_file_path)
            # cv2.imwrite(save_file_path,dst)

        # cv2.imshow('dst', img.dst)
        # cv2.waitKey(0)