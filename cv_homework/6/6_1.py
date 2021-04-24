import os

import numpy as np
import matplotlib.pyplot as plt
import pywt

from utils import draw_picture

"""
作业6：
    在网上寻找一张福州大学校园图像，并将之转换为灰度图像，完成以下题目：
    1编程实现图像的二维快速小波变换（分解到二级尺度）。
    2 利用小波变换后得到的图像，再重构回原来的图像。
    允许直接调用小波变换函数或软件包，将程序代码和实验结果图上传。编程语言不限。
    https://www.icourse163.org/spoc/learn/FZU-1462424162?tid=1463209447#/learn/hw?id=1237676621

参考：
    https://pywavelets.readthedocs.io/en/latest/ref/nd-dwt-and-idwt.html?highlight=wavedecn#pywt.wavedecn
    https://stackoverflow.com/questions/46136454/multilevel-wavelet-decomposition-not-working
    https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html?highlight=dwt_max_level#pywt.dwt_max_level
"""


def main():
    dirname = os.path.dirname(__file__)
    img = plt.imread(f'{dirname}/Fzu_shutong_L.jpg')
    # 过小的图像会导致警告：UserWarning: Level value of 2 is too high: all coefficients will experience boundary effects.
    draw = draw_picture(3, 3)
    plt.figure(figsize=(11, 7))
    draw(1, img, '原图')

    coeffs = pywt.wavedecn(img, wavelet='haar', level=2)
    # <class 'list'> 3: 即level+1
    cAn, *detail_n = coeffs
    # <class 'numpy.ndarray'> [<class 'dict'> ]
    # detail_n.keys(): dict_keys(['ad', 'da', 'dd'])
    rec_img = pywt.waverecn(coeffs, wavelet='haar')

    for i, detail in enumerate(detail_n, start=1):
        draw(1+3*i, detail['ad'], f'{i}级 水平')
        draw(2+3*i, detail['da'], f'{i}级 垂直')
        draw(3+3*i, detail['dd'], f'{i}级 对角')
    draw(3, rec_img, '重建图')
    plt.savefig(f'{dirname}/Fzu_shutong_L_rec.jpg')
    plt.show()


if __name__ == "__main__":
    main()

