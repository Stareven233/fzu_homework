from PIL import Image
import numpy as np
import random


def save_img(img: np.ndarray, filename: str) -> None:
    img = Image.fromarray(img)
    img.save(filename)


def laplace_matrix(img: np.ndarray) -> np.ndarray:
    # g = (1, 1, 1), (1, -8, 1), (1, 1, 1)
    g = (0, 1, 0), (1, -4, 1), (0, 1, 0)
    g = np.asarray(g)
    # 扩展拉普拉斯算子，相邻八个点都考虑
    # laplace_img = np.pad(img, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    img = np.pad(img, 1)
    laplace_img = np.empty_like(img)
    H, W = laplace_img.shape
    for h in range(1, H-1):
        for w in range(1, W-1):
            laplace_img[h, w] = np.sum(img[h-1:h+2, w-1:w+2] * g)
    return laplace_img[1:-1, 1:-1]


image = np.asarray(Image.open("Fzu_shutong_L.jpg"))
laplace_img = laplace_matrix(image)
save_img(laplace_img, "Fzu_shutong_laplace.jpg")
image = image - laplace_img
# 拉普拉斯算子中心系数为负，应减去
save_img(image, "Fzu_shutong_sharp.jpg")
