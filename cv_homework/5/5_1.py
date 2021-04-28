"""
作业5：
    编写一个程序，输入两幅彩色图像和一个二维掩模图像，产生两幅图像混合的拉普拉斯金字塔。
    创建每个图像各自的拉普拉斯金字塔。
    创建两幅掩模图像的高斯金字塔（输入图像和它的补集）。
    将每幅图像乘以对应的掩模，对图像求和。
    从混合的拉普拉斯金字塔中重建最终图像。
https://www.icourse163.org/spoc/learn/FZU-1462424162?tid=1463209447#/learn/hw?id=1237659627
参考：
    https://docs.opencv.org/4.5.1/d4/d86/group__imgproc__filter.html#gaf9bba239dfca11654cb7f50f889fc2ff
    https://docs.opencv.org/4.5.1/d2/de8/group__core__array.html#gaa0f00d98b4b5edeaeb7b8333b2de353b
"""
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import draw_picture


class ImgBlender():
    """
    实现基于拉普拉斯金字塔的图像融合
    """

    def __init__(self, max_level=4):
        """
        参数：
            高斯金字塔层数max_level，也即拉普拉斯金字塔层数-1
        """

        self.max_level = max_level
        # 包括第0层的原图在内
        self.la_pyramid1 = None
        self.la_pyramid2 = None
        self.ga_pyramid_mask1 = None
        self.ga_pyramid_mask2 = None
        self.mixed_img = None

    def generate_ga_pyramid(self, img):
        """
        借助opencv生成高斯金字塔
        默认采用高斯核卷积，再去除偶数行、列
        返回列表：下标越大层次越高，图片越小
        """

        pyramid = [img]
        for i in range(self.max_level-1):
            img = cv2.pyrDown(img)
            pyramid.append(img)
        return pyramid

    def generate_la_pyramid(self, ga_pyramid):
        """
        借助opencv生成拉普拉斯金字塔
        返回列表：下标越大层次越高，图片越小
        """

        pyramid = []
        # 高斯最顶层(最小的)直接作为拉普拉斯
        for i in range(1, self.max_level):
            img = cv2.pyrUp(ga_pyramid[i])
            pyramid.append(ga_pyramid[i-1] - img)
        pyramid.append(ga_pyramid[-1])
        return pyramid

    def run(self, img1, img2, mask):
        """
        参数：
            两幅图像img1, img2，三通道rgb
            掩模图像mask，指定融合部位
        """

        mask = np.where(mask > 127, 1, 0)
        mask = mask.astype(np.uint8)
        # mask 二值化，当需要显示的时候再乘上255

        ga_pyramid1 = self.generate_ga_pyramid(img1)
        ga_pyramid2 = self.generate_ga_pyramid(img2)
        self.la_pyramid1 = self.generate_la_pyramid(ga_pyramid1)
        self.la_pyramid2 = self.generate_la_pyramid(ga_pyramid2)
        # 利用高斯金字塔创建每个图像各自的拉普拉斯金字塔
        self.ga_pyramid_mask1 = self.generate_ga_pyramid(mask)
        self.ga_pyramid_mask2 = self.generate_ga_pyramid(1 - mask)
        # mask2是mask1的补集，分别用于确定两幅图像各自留下的部分
        # 创建两幅掩模图像的高斯金字塔（输入图像和它的补集）
        mixed_la_pyramid = []
        for i in range(self.max_level):
            mixed1 = self.la_pyramid1[i] * self.ga_pyramid_mask1[i]
            mixed2 = self.la_pyramid2[i] * self.ga_pyramid_mask2[i]
            mixed_la_pyramid.append(mixed1 + mixed2)
        # 逐层将拉普拉斯金字塔中每幅图像的乘以对应的掩模，对图像求和
        
        mixed_img = mixed_la_pyramid[-1]
        # 最小的一个没有更上一层的采样相加，故直接使用
        for i in range(self.max_level-2, -1, -1):
            # 从次高层开始加到最低层(原图)
            mixed_img = cv2.pyrUp(mixed_img) + mixed_la_pyramid[i]
        # 从混合的拉普拉斯金字塔中重建最终图像。
        self.mixed_img = mixed_img
        return mixed_img


def main():
    # print("meow")
    # exit()
    dirname = os.path.dirname(__file__) + '/usagi'
    img1 = plt.imread(f'{dirname}/img1.jpg')
    img2 = plt.imread(f'{dirname}/img2.jpg')
    mask = plt.imread(f'{dirname}/mask.jpg')
    # cv2.imread读取的也是ndarray，却不能处理带中文的路径

    # mask = np.dsplit(mask, 3)
    # np.savetxt('mask1', mask[0].squeeze(), fmt='%.1f')
    # np.savetxt('mask2', mask[1].squeeze(), fmt='%.1f')
    # np.savetxt('mask3', mask[2].squeeze(), fmt='%.1f')

    blender = ImgBlender(max_level=2)
    # 图像尺寸最好为2的幂次，当level高可能导致边缘出现奇异色块
    mixed_img = blender.run(img1, img2, mask)
    draw = draw_picture(2, 3)
    plt.figure(figsize=(9, 6))
    draw(1, img1, 'img1')
    draw(2, img2, 'img2')
    draw(3, mask, 'mask')
    draw(4, mixed_img, 'mixed')
    cropped_img = img1*blender.ga_pyramid_mask1[0] + img2*blender.ga_pyramid_mask2[0]
    # 取0-1化的mask切分图像
    draw(6, cropped_img, 'cropped')

    plt.savefig('rabbit_in_fzu.jpg')
    plt.show()
    draw = draw_picture(1, 1)
    plt.figure(figsize=(6, 6))
    draw(1, mixed_img, 'mixed')
    plt.savefig('mixed_img.jpg')
    plt.figure(figsize=(6, 6))
    draw(1, cropped_img, 'cropped')
    plt.savefig('cropped_img.jpg')


if __name__ == "__main__":
    main()

