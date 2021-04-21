from PIL import Image
import numpy as np


def save_img(img: np.ndarray, filename: str) -> None:
    img = Image.fromarray(img)
    img.save(filename)


def normalization(img: np.ndarray) -> np.ndarray:
    img_min = np.min(img)
    range_ = np.max(img) - img_min
    return (img - img_min) / range_


image = np.asarray(Image.open("Fzu_shutong_L.jpg"))

# 反转变换
image_inv = np.max(image) - image
save_img(image_inv, "Fzu_shutong_inverted.jpg")

# 对数变换
image_log = 1 * np.log(1 + image.astype(np.float16))
image_log = 255 * normalization(image_log)
image_log = image_log.astype(np.uint8)
save_img(image_log, "Fzu_shutong_log.jpg")

# 幂变换
image_pow = 1 * image**1.6
# 幂次越小越亮
image_pow = 255 * normalization(image_pow)
image_pow = image_pow.astype(np.uint8)
save_img(image_pow, "Fzu_shutong_pow.jpg")
