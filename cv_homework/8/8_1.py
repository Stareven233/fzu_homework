import numpy as np

"""
作业8：
    针对一张20*20的图像块，编程计算该图像块的灰度共生矩阵（d=1,θ=0°），
    并将原始图像块和共生矩阵的数值显示出来。
    将程序代码也上传。编程语言不限。
    https://www.icourse163.org/spoc/learn/FZU-1462424162?tid=1463209447#/learn/hw?id=1237729587

参考：
    https://www.icourse163.org/spoc/learn/FZU-1462424162?tid=1463209447#/learn/content?type=detail&id=1242844232&cid=1266035646
"""


def glcm(arr=None, arr_size=(20, 20), gray_level=8):
    """
    灰度共生矩阵：Gray-Level Co-occurrence Matrix
    d=1, θ=0°

    随机生成一个20*20的像素块，其中值为0-20的整数
    则灰度共生矩阵为20*20的数组

    实际上
    图像为M*N，灰度级别为G，即像素共有G个不同的值
    灰度共生矩阵为G*G
    """

    if arr is None:
        mu, sigma = gray_level//4, gray_level//2
        arr = np.random.normal(loc=sigma, scale=mu, size=arr_size)
        # 95.45% 在 μ+-2σ 范围内
        # 返回一个由size指定形状的数组，数组中的值服从 μ=loc, σ=scale 的正态分布
        arr = arr.astype(dtype=np.uint8)
        arr = np.clip(arr, 0, gray_level-1)
        # np.set_printoptions(suppress=True)
    else:
        arr = np.asarray(arr)

    res = np.zeros((gray_level, gray_level), dtype=np.uint8)
    for row in arr:
        for i in range(gray_level - 1):
            a, b = row[i], row[i+1]
            res[a, b] += 1
            res[b, a] += 1
    return arr, res

def main():
    # arr = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 2, 2, 2], [2, 2, 3, 3]]
    # a, g = glcm(arr, gray_level=4)
    # print("示例图像块\n", a, end="\n\n")
    # print("对应灰度共生矩阵\n", g)
    
    a, g = glcm()
    print("随机图像块\n", a, end="\n\n")
    print("对应灰度共生矩阵\n", g)


if __name__ == "__main__":
    main()
