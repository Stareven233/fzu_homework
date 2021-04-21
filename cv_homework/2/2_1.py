from PIL import Image
import numpy as np
import random


def save_img(img: np.ndarray, filename: str) -> None:
    img = Image.fromarray(img)
    img.save(filename)


def salt_and_pepper_noise(img: np.ndarray, proportion: int=0.05) -> np.ndarray:
    noise_img = img.copy()
    height, width = noise_img.shape[0], noise_img.shape[1]
    num = int(height * width * proportion)
    # 多少个像素点添加椒盐噪声
    for i in range(num):
        w = random.randint(0, width - 1)
        h = random.randint(0, height - 1)
        if random.randint(0, 1) == 0:
            noise_img[h, w] = 0
        else:
            noise_img[h, w] = 255
    return noise_img


def median_filter(img: np.ndarray, FILTER_SIZE: int=3) -> np.ndarray:
    IMG_MARGIN = (FILTER_SIZE - 1)//2
    # 仅奇数
    H, W = img.shape
    res_img = img.copy()
    for h in range(IMG_MARGIN, H-IMG_MARGIN):
      for w in range(IMG_MARGIN, W-IMG_MARGIN):
        res_img[h, w] = np.median(res_img[h-IMG_MARGIN:h+IMG_MARGIN+1, w-IMG_MARGIN:w+IMG_MARGIN+1])
    return res_img


FILTER_SIZE = 3
image = np.asarray(Image.open("Fzu_shutong_L.jpg"))
noise_img = salt_and_pepper_noise(image)
save_img(image, "Fzu_shutong_noise.jpg")
restored_image = median_filter(image, 3)
save_img(restored_image, "Fzu_shutong_median_filter.jpg")


