import struct

import numpy as np
import matplotlib.pyplot as plt


class MNIST:
    def __init__(self, data_path):
        self.mnist_path = data_path
        self.train_data, self.train_labels = self.__load('train')
        self.test_data, self.test_labels = self.__load('t10k')

    def __load(self, kind='t10k'):
        data_path = f'{self.mnist_path}/{kind}-images.idx3-ubyte'
        label_path = f'{self.mnist_path}/{kind}-labels.idx1-ubyte'

        with open(label_path, 'rb') as f:
            # magic, n = struct.unpack('>II', f.read(8))
            f.read(8)
            # 读取前8个字节（控制信息），这部分不属于图像数据，应排除
            labels = np.fromfile(f, dtype=np.uint8)
        with open(data_path, 'rb') as f:
            # magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            f.read(16)
            images = np.fromfile(f, dtype=np.uint8).reshape(len(labels), -1)
            # 若不reshape：val_lables：(10000,) val_data：(7840000,)
            # mnist是灰度图像，每张图片(28, 28, 1)
        return images, labels
    
    def show(self, num=20):
        fig = plt.figure(figsize=(8, 5))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        samples = np.random.choice(len(self.train_data), num)
        # 随机抽几张看看效果
        for seq, i in enumerate(samples):
            images = np.reshape(self.train_data[i], (28, 28))
            ax = fig.add_subplot(6, 5, seq+1, xticks=[], yticks=[])
            ax.imshow(images, cmap=plt.cm.binary, interpolation='nearest')
            ax.text(1, 6, str(self.train_labels[i]))
        plt.show()


def load_images(file_name):
    binfile = open(file_name, 'rb') 
    # file object = open(file_name [, access_mode][, buffering]) 
    # buffering 0表示不使用缓冲，1表示在访问一个文件时进行缓冲。     
    buffers = binfile.read()
    magic, num, rows, cols = struct.unpack_from('>IIII', buffers, 0)
    # 读取image文件前4个整型数字，这是struct库打包时写入的描述信息
    # struct库参考https://zhuanlan.zhihu.com/p/35856929
    bits = num * rows * cols
    # 整个images数据大小为60000*28*28
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    # 关闭文件
    binfile.close()
    # 转换为[60000,784]型数组
    images = np.reshape(images, [num, rows * cols])
    return images


def load_labels(file_name):
    binfile = open(file_name, 'rb')
    buffers = binfile.read()
    magic, num = struct.unpack_from('>II', buffers, 0) 
    # 读取label文件前2个整形数字，label的长度为num
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    binfile.close()
    labels = np.reshape(labels, [num])
    return labels
