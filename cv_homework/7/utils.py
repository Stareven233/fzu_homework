import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def save_array_img(img: np.ndarray, filename: str) -> None:
    img = Image.fromarray(img)
    img.save(filename)


def log_img(image):
    return np.log(1 + np.abs(image))


def normalization(mat: np.ndarray) -> np.ndarray:
    img_min = np.min(mat)
    range_ = np.max(mat) - img_min
    return (mat - img_min) / range_


def draw_picture(row: int, col: int, f_size: tuple):
    r = row
    c = col
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=f_size)
    # plt.figure(figsize=(c*3, r*1+1))

    def inner(idx, pic, title, axis_on="off", cmap='gray'):
        ax = plt.subplot(r, c, idx)
        ax.set_title(title, fontsize=10)
        ax.axis(axis_on)
        ax.set_xticks([])
        ax.set_yticks([])
        if cmap is None:
            plt.imshow(pic)
        else:
            plt.imshow(pic, cmap=cmap)
    return inner


def save_picture(row: int=1, col: int=1, f_size: tuple=(12, 12)):
    ax = plt.subplot(row, col, 1)
    ax.axis("off")

    def inner(img: np.ndarray, path: str, color_map=None):
        plt.figure(figsize=f_size)
        plt.imshow(img, cmap=color_map)
        plt.xticks([])  #去掉x轴
        plt.yticks([])  #去掉y轴
        plt.axis('off')  #去掉坐标轴
        # 必须在imshow调用之后
        plt.savefig(path)
    return inner


def draw_scatter(row: int, col: int):
    r = row
    c = col
    plt.figure(figsize=(14, 8))
    
    def inner(idx, x, y, fsize=6, color="#ff5050"):
        x_d, x_label = x
        y_d, y_label = y
        plt.subplot(r, c, idx)
        plt.xlabel(x_label)  # x轴名称
        plt.ylabel(y_label)  # y轴名称
        plt.grid(True)  # 显示网格线
        plt.scatter(x_d, y_d, s=fsize, c=color)
    return inner


def draw_decision_border(row: int, col: int):
    r = row
    c = col
    plt.figure(figsize=(14, 8))

    def get_border(cls_proto):
        a = cls_proto[0] - cls_proto[1]
        a = -a[0] / (a[1] + 1e-8)
        # 根据 k1*k2 = -1 求两点间垂线
        mid = np.mean(cls_proto, axis=0)
        b = mid[1] - a*mid[0]
        return a, b


    def inner(idx, x, y, x_proto, y_proto, fsize=6, color="#ff5050"):
        """
        x: 一列属性(n, 1)
        y: 一列属性(n, 1)
        x_proto: 两个类的原型(4, )vstack后(2, 4)其中的一列(2, 1)
        y_proto: 两个类的原型(4, )vstack后(2, 4)其中另一列(2, 1)
        """

        x_d, x_label = x
        y_d, y_label = y
        plt.subplot(r, c, idx)
        plt.xlabel(x_label)  # x轴名称
        plt.ylabel(y_label)  # y轴名称
        plt.grid(True)  # 显示网格线
        plt.scatter(x_d, y_d, s=fsize, c=color)
        plt.scatter(x_proto, y_proto, s=16, c="#f19d63")
        cls_proto = np.hstack((x_proto, y_proto))
        a, b = get_border(cls_proto)
        border_x = np.linspace(np.min(x_d), np.max(x_d), 60)
        border_y = a*border_x + b
        r_min = min(np.min(x_d), np.min(y_d))
        r_max = max(np.max(x_d), np.max(y_d))
        plt.xlim((r_min, r_max))   # 设置x轴的范围
        plt.ylim((r_min, r_max))   # 设置y轴的范围
        # plt.xticks(np.linspace(r_min, r_max+1, 6))
        # plt.yticks(np.linspace(r_min, r_max+1, 6))
        plt.plot(border_x, border_y, color="#62a973", linewidth=1.2)
    return inner
