import numpy as np
import matplotlib.pyplot as plt

from utils import draw_picture

"""
作业4：
    肤色检测
    基于色度或其它彩色属性设计一个简单的肤色检测器。
    用手机自拍一张人脸图像。
    剪切照片或另外用画笔工具拾取那些可能是肤色的像素（人脸部位）。
    对这些肤色像素计算彩色分布，如彩色直方图。
    对背景像素计算彩色分布。
    用手机再自拍一张人脸图像，试着用这两个分布函数，在新拍图像中寻找肤色区域。
https://www.icourse163.org/spoc/learn/FZU-1462424162?tid=1463209447#/learn/hw?id=1237614776
"""


class SkinDetector():
    """
    传入三通道rgb图像的肤色部分及背景部分图像，检测器学习后
    再传入完整图像进行检测
    """

    def __init__(self, hist_bins=64, threshhold_factor=0.3):
        self.h_bins = hist_bins
        # 计算直方图时将256分成几份
        self.bin_size = 256 // hist_bins
        self.skin_hist = None  # np.random.randint(0, hist_bins, size=(3, 256))
        self.bg_hist = None
        self.skin_th = None
        self.bg_th = None
        self.th_factor = threshhold_factor

    def _hist(self, img):
        # img = np.sort(img.ravel())
        # nan_idx = np.argmax(img)
        hist =  np.histogram(img.ravel(), bins=self.h_bins, range=[0, 256], density=True)
        # range: Values outside the range are ignored. 没必要特意去掉非法像素点
        return hist[0]
        # (64, )
        # 只要统计的区间概率，不需要区间
        # 用map调用该函数看不到输出

    @staticmethod
    def _preprocess(img):
        """
        rgba换成rgb(整型)，同时alpha不为1的丢弃
        """

        *rgb, a = np.dsplit(img, 4)
        rgb = np.asarray(rgb)
        rgb *= 256
        rgb = rgb.astype(np.uint8).squeeze()
        a = a.astype(np.uint8).squeeze()
        # 不进行squeeze会导致：operands could not be broadcast together with shapes (352,256,1) (3,352,256) ()
        rgb = np.where(a == 1, rgb, 666)
        # 666超出256，表示该像素丢弃，不参与计算直方图
        # 根据第三个参数的类型会影响返回值类型，若这里用NaN就会变成float64
        return rgb

    def train(self, skin_img, bg_img):
        """
        skin_img, bg_img皆为浮点rgba格式，alpha=1表示保留
        """

        rgb_chans = self._preprocess(skin_img)
        # self._hist(rgb_chans[0])
        self.skin_hist = np.asarray(list(map(self._hist, rgb_chans)))
        # self.skin_hist = np.asarray(map(self._hist, rgb_chans))
        # 这样会导致返回值shape=()，真的就是一对括号
        self.skin_th = np.max(self.skin_hist, axis=1) * self.th_factor
        # 设axis=i，则Numpy沿着第i个下标变化的方向进行操作
        rgb_chans = self._preprocess(bg_img)
        self.bg_hist = np.asarray(list(map(self._hist, rgb_chans)))
        self.bg_th = np.max(self.bg_hist, axis=1) * self.th_factor

    def run(self, img):
        img_h, img_w = img.shape[:2]
        img = img.astype(np.uint8)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for h in range(img_h):
            for w in range(img_w):
                f = True
                # 该像素是否属于皮肤部分
                p_bins = img[h, w, :] // self.bin_size
                # 该像素三通道分别所属的直方图区域
                skin_prop = self.skin_hist[np.arange(3), p_bins]
                bg_prop = self.bg_hist[np.arange(3), p_bins]
                f = f and np.alltrue(skin_prop > self.skin_th)
                # 该像素三通道分别对应各自直方图的比例
                f = f and np.alltrue(bg_prop < self.bg_th)
                f = f and np.alltrue(skin_prop > bg_prop)
                mask[h, w] = 255 if f else 0
        return mask, np.expand_dims(mask, axis=2)&img  
        # (h, w) -> (h, w, 1) 才能合 (h, w, 3) 运算
        # np.bitwise_and(mask, img)


def main():
    detector = SkinDetector(32, 0.29)
    # 这图背景较易区分，故可设hist_bins更低有助于选择更多的人脸部分
    skin = plt.imread('./4/xwg_skin.png')
    bg = plt.imread('./4/xwg_bg.png')
    detector.train(skin, bg)
    
    img = plt.imread('./4/xwg2.jpg')
    mask, skin_img = detector.run(img)

    draw = draw_picture(2, 3)
    plt.figure(figsize=(9, 6))
    draw(1, skin, 'train_skin')
    draw(3, bg, 'train_bg')
    draw(4, img, '原图')
    draw(5, mask, 'mask')
    draw(6, skin_img, '皮肤部分')
    plt.savefig('./4xwg_detected.jpg')
    plt.show()


if __name__ == "__main__":
    main()

