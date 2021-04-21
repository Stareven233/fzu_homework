from PIL import Image
import numpy as np


def save_img(img: np.ndarray, filename: str) -> None:
    img = Image.fromarray(img)
    img.save(filename)


image = np.asarray(Image.open("Fzu_shutong_L.jpg"))

# 直方图均衡化
prob = np.zeros(shape=256)
for pixel in image.flatten():
    prob[pixel] += 1
r, c = image.shape
prob = prob / (r * c)
prob = np.cumsum(prob)
# 计算像素值累积概率密度分布

# img_map = [int(i * prob[i]) for i in range(256)]
img_map = prob * 255
# 各像素值均衡后的值映射
image_hist = np.empty_like(image)
r, c = image_hist.shape
for ri in range(r):
    for ci in range(c):
        image_hist[ri, ci] = img_map[image[ri, ci]]
save_img(image_hist, f"Fzu_shutong_hist.jpg")
