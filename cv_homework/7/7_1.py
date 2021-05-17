import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils import draw_picture
from utils import save_picture

"""
作业7：
    在网上寻找一张福州大学校园图像，并将之转换为灰度图像，完成以下题目：
    1选用某个边缘提取算子，编程实现图像边缘轮廓提取。
    2分析角点特征，编程实现图像角点提取。
    将程序代码和实验结果图上传。编程语言不限。
    https://www.icourse163.org/spoc/learn/FZU-1462424162?tid=1463209447#/learn/hw?id=1237694144

参考：
    https://docs.opencv.org/4.5.1/d4/d86/group__imgproc__filter.html#gaa13106761eedf14798f37aa2d60404c9
    https://docs.opencv.org/4.5.1/df/d74/classcv_1_1FastFeatureDetector.html
"""


def main():
    dirname = os.path.dirname(__file__)
    img = plt.imread(f'{dirname}/Fzu_shutong_L.jpg')
    draw = draw_picture(1, 3, (15, 4))
    draw(1, img, '原图')

    dx = cv2.Scharr(img, -1, 1, 0)
    dy = cv2.Scharr(img, -1, 0, 1)
    d_img = cv2.addWeighted(dx, 0.5, dy, 0.5, 0)
    draw(2, d_img, 'Scharr边缘')

    fast = cv2.FastFeatureDetector_create(threshold=42, nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_7_12)
    keypoints = None
    keypoints = fast.detect(img)
    corner_img = cv2.drawKeypoints(img, keypoints, None, color=(255, 142, 29))
    draw(3, corner_img, 'Fast角点')
    plt.savefig(f'{dirname}/Fzu_shutong_L_ed.jpg')
    plt.show()

    save = save_picture()
    save(d_img, f'{dirname}/Scharr边缘.jpg', "gray")
    save(corner_img, f'{dirname}/Fast角点.jpg')

if __name__ == "__main__":
    main()

