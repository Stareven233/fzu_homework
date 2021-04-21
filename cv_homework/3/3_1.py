import numpy as np
import matplotlib.pyplot as plt

from utils import draw_picture

"""

作业3：
    在网上寻找一张福州大学校园图像，并将之转换为灰度图像，完成以下题目：
    1编程实现陷波滤波器，对该图进行频率域滤波。
    2编程实现巴特沃思低通滤波器，对该图进行图像滤波。
    3编程实现理想低通滤波器，对该图进行图像滤波，并分析一下振铃现象。
    将程序代码和实验结果图上传。编程语言不限。
https://www.icourse163.org/spoc/learn/FZU-1462424162?tid=1463209447#/learn/content?type=detail&id=1242343011&cid=1265276221
"""


def log_img(image):
    return np.log(1 + np.abs(image))


class Filter():
    def __init__(self, img):
        self.img = img

    def _notch_kernel(self, d0=1000):
        """
        理想陷波带阻滤波器
        """

        r, c = self.img.shape
        u0, v0 = 0, c/8
        h = np.empty((r, c, ))
        for u in range(r):
            for v in range(c):
                d1 = (u-r/2-u0)**2 + (v-c/2-v0)**2
                d2 = (u-r/2+u0)**2 + (v-c/2+v0)**2
                # 挖去的两个点偏移了(u0, v0)
                h[u, v] = int(min(d1, d2) >= d0)
        return h
    
    def _butterworth_kernel(self, d0=42):
        """
        巴特沃斯低通滤波器
        """

        r, c = self.img.shape
        n = 2
        h = np.empty((r, c, ))
        k = n<<1
        for u in range(r):
            for v in range(c):
                d = np.sqrt((u - r/2)**2 + (v - c/2)**2)
                # 挖去的两个点偏移了(u0, v0)
                h[u, v] = 1 / (1 + d/d0)**k
        return h
    
    def _ideal_lowpass_kernel(self, d0=33):
        """
        理想低通滤波器
        """

        r, c = self.img.shape
        # d0 = 33  # 35, 36分界线
        h = np.empty((r, c, ))
        for u in range(r):
            for v in range(c):
                d = np.sqrt((u - r/2)**2 + (v - c/2)**2)
                # 挖去的两个点偏移了(u0, v0)
                h[u, v] = int(d <= d0)
        return h

    def run(self, method: str) -> tuple:
        kernel = f'_{method}_kernel'
        if not hasattr(self, kernel):
            raise NotImplementedError('不存在该种滤波器')
        f = np.fft.fft2(self.img)
        f_shift = np.fft.fftshift(f)
        # f_shift_h = f_shift * self._notch_kernel()
        f_shift_h = f_shift * getattr(self, kernel)()
        f_h = np.fft.ifftshift(f_shift_h)
        img_h = np.fft.ifft2(f_h)
        return f_shift, f_shift_h, img_h
    
    def ringing_effect(self, d0: int):
        f_shift_h = self._ideal_lowpass_kernel(d0)
        f_h = np.fft.ifftshift(f_shift_h)
        h = np.fft.ifft2(f_h)
        return f_shift_h, h


def main():
    img = plt.imread('boy_L.jpg')
    # 对带噪声的图片滤波效果更加明显，有针对性
    # img = plt.imread('Fzu_shutong_L.jpg')
    filters = Filter(img)
    # draw = draw_picture(2, 2)

    # plt.figure(figsize=(10, 6))
    # f_shift, f_shift_h, img_h = filters.run('notch')
    # draw(1, img, '原图')
    # draw(2, log_img(f_shift), '频率域')
    # draw(3, log_img(f_shift_h), 'NF滤波后频率域')
    # draw(4, log_img(img_h), 'NF滤波后空间域')
    # plt.savefig('Fzu_shutong_L_notch.jpg')
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # f_shift, f_shift_h, img_h = filters.run('butterworth')
    # draw(1, img, '原图')
    # draw(2, log_img(f_shift), '频率域')
    # draw(3, log_img(f_shift_h), 'BLPF滤波后频率域')
    # draw(4, log_img(img_h), 'BLPF滤波后空间域')
    # plt.savefig('Fzu_shutong_L_butterworth.jpg')
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # f_shift, f_shift_h, img_h = filters.run('ideal_lowpass')
    # draw(1, img, '原图')
    # draw(2, log_img(f_shift), '频率域')
    # draw(3, log_img(f_shift_h), 'ILPF滤波后频率域')
    # draw(4, log_img(img_h), 'ILPF滤波后空间域')
    # plt.savefig('Fzu_shutong_L_ideal_lowpass.jpg')
    # plt.show()

    draw = draw_picture(1, 2)
    plt.figure(figsize=(10, 3))
    fh, h = filters.ringing_effect(d0=3)
    draw(1, log_img(fh), '频率域滤波器')
    draw(2, log_img(h), '空间域滤波器')
    # without log_img: TypeError: Image data of dtype complex128 cannot be converted to float
    plt.savefig('boy_L_ringing_effect.jpg')
    plt.show()


if __name__ == "__main__":
    main()
    """
    振铃现象：
    与滤波器H大小、截止频率d0均有关联
    当d0较小(个位数)方能观察到，本图片里当d0=3最为明显
    振铃现象处理的图片有较大扭曲变形
    """
