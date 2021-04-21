import numpy as np
import matplotlib.pyplot as plt
import pylab


img = plt.imread('Fzu_shutong_L.jpg')
# plt.figure(dpi=160)
plt.figure(figsize=(16, 9))

plt.subplot(231)
# 两行三列，位置是1的子图
plt.imshow(img, 'gray')
plt.title('original')
#进行傅立叶变换，并显示结果

fft2 = np.fft.fft2(img)
plt.subplot(2, 3, 2)
plt.imshow(np.abs(fft2), 'gray')
plt.title('fft2')
#将图像变换的原点移动到频域矩形的中心，并显示效果

shift2center = np.fft.fftshift(fft2)
plt.subplot(233)
plt.imshow(np.abs(shift2center), 'gray')
plt.title('shift2center')
#对傅立叶变换的结果进行对数变换，并显示效果

log_fft2 = np.log(1 + np.abs(fft2))
plt.subplot(234)
plt.imshow(log_fft2, 'gray')
plt.title('log_fft2')
#对中心化后的结果进行对数变换，并显示结果

log_shift2center = np.log(1 + np.abs(shift2center))
plt.subplot(235)
plt.imshow(log_shift2center, 'gray')
plt.title('log_shift2center')
plt.savefig('Fzu_shutong_L_fft.jpg')
pylab.show()
