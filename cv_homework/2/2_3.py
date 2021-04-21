import numpy as np
import matplotlib.pyplot as plt
import pylab


img = plt.imread('Fzu_shutong_L.jpg')
# plt.figure(dpi=160)
plt.figure(figsize=(12, 4.5))
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

plt.subplot(121)
# 一行两列，位置是1的子图
plt.imshow(img, 'gray')
plt.title('原图')
#进行傅立叶变换，并显示结果

fft2 = np.fft.fft2(img)
shift2center = np.fft.fftshift(fft2)
log_shift2center = np.log(1 + np.abs(shift2center))

plt.subplot(1, 2, 2)
plt.imshow(log_shift2center, 'gray')
plt.title('频域图_shift_log')
plt.savefig('Fzu_shutong_L_fft.jpg')
pylab.show()
