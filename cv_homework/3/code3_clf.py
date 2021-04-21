import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Im

figure = plt.figure()
n = 2
m = 2
plt.subplot(n, m, 1)


def normalize(data):
    tmp = np.array(data)
    minimum = np.min(tmp)
    maximum = np.max(tmp)
    return (tmp - minimum)/(maximum - minimum)


def paste(seq, pic, pic_title, axis_on_off="off", model=None):
    ax = plt.subplot(n, m, seq)
    ax.axis(axis_on_off)
    ax.set_title(pic_title)
    if model != None:
        plt.imshow(pic, cmap=model)
    else:
        plt.imshow(pic)
    return


def bar_table_paste(seq, x, y, pic_title, axis_on_off="on"):
    ax = plt.subplot(n, m, seq)
    ax.axis(axis_on_off)
    ax.set_title(pic_title)
    plt.bar(x, y)
    return


def equalization(data):
    cnt_result = np.bincount(data.reshape((-1)), minlength=256)
    tot = np.sum(cnt_result)
    cnt_result = cnt_result / tot
    T = (255 * np.array([np.sum(cnt_result[:idx]) for idx in range(1, 257)])).astype("uint8")
    return cnt_result, T[data], np.bincount(T[data].reshape((-1)), minlength=256) / tot # np.array([T[r] for r in data])


class picture():
    
    def __init__(self, pic_path, pic_name):
        self.read_path = pic_path + pic_name
        self.data = Im.open(self.read_path)
        self.image_format = self.data.format
        self.col, self.row = self.data.size
        self.save_path = ''
        self.tmp_data = None
    
    def save(self, save_name):
        Im.fromarray(self.tmp_data.astype("uint8")).save(self.save_path + save_name  + '.' + self.image_format)

    def convert_to_gray(self, direct_convert):
        '''
        paste(1, self.data, "origin")
        '''
        if direct_convert:
            self.tmp_data = np.array(self.data.convert("L"))
        else:
            self.tmp_data = np.mean(np.array(self.data), axis=2)
        '''
        paste(2, self.tmp_data, "gray", model=plt.cm.gray)
        '''
        return

    def reverse(self):
        self.tmp_data = 255 - np.array(self.data)
        paste(3, self.tmp_data, "reverse")
        return

    def log_transform(self, c=1.0):
        self.tmp_data = c * np.log10( 1+np.array(self.data).astype("float") )
        self.tmp_data = ( 254 * normalize(self.tmp_data) ).astype("uint8")
        paste(4, self.tmp_data, "log_tf")
        return

    def power_transform(self, c=1.0, gamma=1.0):
        def detect(c, gamma):
            if c <= 0 or gamma <= 0:
                raise Exception("Error! the params [c=%f, gamma=%f] need to be all positive!" %
                                ( c, gamma) )
        detect(c, gamma)

        self.tmp_data = c * np.power(np.array(self.data).astype("float"), gamma)
        self.tmp_data = ( 254 * normalize(self.tmp_data) ).astype("uint8")
        paste(5, self.tmp_data, "power_tf [%s, gamma%s]" % (
                     ("lighter", "<1") if gamma<1.0 else ("darker", ">1")))

        gamma = 2.0 - gamma
        self.tmp_data = c * np.power(np.array(self.data).astype("float"), gamma)
        self.tmp_data = ( 254 * normalize(self.tmp_data) ).astype("uint8")
        paste(6, self.tmp_data, "power_tf [%s, gamma%s]" % (
                     ("lighter", "<1") if gamma<1.0 else ("darker", ">1")))
        return

    def histogram_equalization(self):
        self.tmp_data = np.array(self.data)
        red, green, blue = self.tmp_data[:, :, 0], self.tmp_data[:, :, 1], self.tmp_data[:, :, 2]
        e_red, e_green, e_blue = equalization(red), equalization(green), equalization(blue)

        bar_table_paste(1, range(256), e_red[0], "red")
        bar_table_paste(2, range(256), e_blue[0], "blue")
        bar_table_paste(3, range(256), e_green[0], "green")
        bar_table_paste(4, range(256), e_red[2], "new_red")
        bar_table_paste(5, range(256), e_blue[2], "new_blue")
        bar_table_paste(6, range(256), e_green[2], "new_green")
        plt.show()

        new_pic = self.tmp_data = np.stack((e_red[1], e_green[1], e_blue[1]), axis=2)
        paste(1, self.data, "origin")
        paste(2, new_pic, "RGB_equalization")
        self.save("RGB_equalization")
        
        
        self.tmp_data = np.array(self.data.convert("L"))
        paste(3, self.tmp_data, "gray", model=plt.cm.gray)
        
        e_gray = equalization(self.tmp_data)
        bar_table_paste(4, range(256), e_gray[0], "gray")
        bar_table_paste(5, range(256), e_gray[2], "new_gray")
        
        paste(6, e_gray[1], "gray_equalization", model=plt.cm.gray)
        self.save("gray_equalization")
        plt.show()
        return

    def median_filter(self):
        r, c = np.array(self.data).shape
        input_pic = np.pad(self.data, (1, 1), mode='constant', constant_values=0)
        output = np.zeros((r, c))

        for _r in range(r):
            for _c in range(c):
                output[_r][_c] = sorted(input_pic[_r:_r+3, _c:_c+3].reshape(-1))[5]
        return output
    
    def laplace_filter(self):
        laplacian_mask = np.array(
                    [[0, 1, 0],
                     [1,-4, 1],
                     [0, 1, 0]]
                    )   
        input_pic = np.pad(self.data, (1, 1), mode='constant', constant_values=0)
        output = np.zeros(np.array(self.data).shape)

        for r in range(output.shape[0]):
            for c in range(output.shape[1]):
                output[r][c] = (np.sum(laplacian_mask * input_pic[r:r+3, c:c+3]))
        return output, output+np.array(self.data)

    def Fourier_transform_2_dimensions(self, low_pass=False, high_pass=False, pass_range=30):
        self.tmp_data = np.array(self.data)

        f_pic = np.fft.fft2(self.tmp_data)
        fftshift_pic = np.fft.fftshift(f_pic)

        row, col = self.tmp_data.shape
        r, c = row//2, col//2
        n = pass_range

        if high_pass:
            fftshift_pic[r-n:r+n, c-n:c+n] = 0.1
            ishift = np.fft.ifftshift(fftshift_pic)

        ishift = np.fft.ifftshift(fftshift_pic)
        iimg = np.fft.ifft2(ishift)
        iimg = np.abs(iimg)
        
        return self.tmp_data, iimg, np.fft.fftshift(f_pic), fftshift_pic

    def notch_filter(self, D0, U0, V0, model="ideal", band_elimitation=False):
        def dis2(u, v, u0, v0):
            return (u-u0)**2+(v-v0)**2

        def H(D0, D1, D2, _model="ideal", n=2):
            if model == "ideal":
                return 0 if D1 <= D0 or D2 <= D0 else 1
            elif model == "guassian":
                return 1 - np.exp(-D1 * D2 / D0**2 / 2)
            elif model == "butterworth":
                return 1 / (1 + ( D0**2 / (D1*D2 + 0.0001) )**n )
            else:
                raise Exception("The notch filter has no model to select!\n")

        self.tmp_data = np.array(self.data)
        f_pic = np.fft.fft2(self.tmp_data)
        fftshift_pic = np.fft.fftshift(f_pic)

        row, col = self.tmp_data.shape
        r, c = row//2, col//2

        H_matrix = np.zeros((row, col))
        for _r in range(row):
            for _c in range(col):
                H_matrix[_r][_c] = H( D0, D1=dis2(_r, _c, r-U0, c-V0), D2=dis2(_r, _c, r+U0, c+V0), _model = model )
                if band_elimitation:
                    H_matrix[_r][_c] = 1 - H_matrix[_r][_c]

        fftshift_pic *= H_matrix
        ishift=np.fft.ifftshift(fftshift_pic)
        iimg = np.fft.ifft2(ishift)
        iimg = np.abs(iimg)

        return self.tmp_data, iimg, np.fft.fftshift(f_pic), fftshift_pic

    def butterworth_filter(self, n, D0):
        def dis(u, v, u0, v0):
            return ((u-u0)**2+(v-v0)**2)**0.5

        def H(n, D0, D):
            return 1/(1+(D/D0)**(2*n))

        self.tmp_data = np.array(self.data)
        f_pic = np.fft.fft2(self.tmp_data)
        fftshift_pic = np.fft.fftshift(f_pic)

        row, col = self.tmp_data.shape
        r, c = row//2, col//2

        H_matrix = np.zeros((row, col))
        for _r in range(row):
            for _c in range(col):
                H_matrix[_r][_c] = H(n, D0, dis(_r, _c, r, c))
                
        fftshift_pic *= H_matrix
        ishift=np.fft.ifftshift(fftshift_pic)
        iimg = np.fft.ifft2(ishift)
        iimg = np.abs(iimg)

        return self.tmp_data, iimg, np.fft.fftshift(f_pic), fftshift_pic

    def ideal_lowpass_filter(self, R):
        def dis(u, v, u0, v0):
            return ((u-u0)**2+(v-v0)**2)**0.5
        
        self.tmp_data = np.array(self.data)
        f_pic = np.fft.fft2(self.tmp_data)
        fftshift_pic = np.fft.fftshift(f_pic)

        row, col = self.tmp_data.shape
        r, c = row//2, col//2

        for _r in range(row):
            for _c in range(col):
                if dis(_r, _c, r, c) > R:
                    fftshift_pic[_r][_c] = 0
        ishift=np.fft.ifftshift(fftshift_pic)
        iimg = np.fft.ifft2(ishift)
        iimg = np.abs(iimg)

        return self.tmp_data, iimg, np.fft.fftshift(f_pic), fftshift_pic
        

def main():
    pic_path = r"D:\A日常\大学\计算机视觉\homework\\"
    pic_name = "boy_L.jpg"
    pic = picture(pic_path, pic_name)

    pic.convert_to_gray(True)
    pic.save('gray')
    pic = picture(pic_path, "gray.JPEG")
    
    # task 1
    # origin, nf_pic, fs_pic, nf_fs_pic = pic.notch_filter(D0=1200, U0=0, V0=50, model="ideal")
    # paste(1, origin, "origin", model=plt.cm.gray)
    # paste(2, nf_pic, "notch_filter", model=plt.cm.gray)
    # paste(3, np.log(np.abs(fs_pic)), "fftshift_frequency_pic", model=plt.cm.gray)
    # paste(4, np.log(np.abs(nf_fs_pic)), "fs_nf_frequency_pic", model=plt.cm.gray)
    # plt.show()
    
    # task 2
    # origin, bwf_pic, fs_pic, bwf_fs_pic = pic.butterworth_filter(n=2, D0=18)
    # paste(1, origin, "origin", model=plt.cm.gray)
    # paste(2, bwf_pic, "butterworth_filter", model=plt.cm.gray)
    # paste(3, np.log(np.abs(fs_pic)), "fftshift_frequency_pic", model=plt.cm.gray)
    # paste(4, np.log(np.abs(bwf_fs_pic)), "fs_bwf_frequency_pic", model=plt.cm.gray)
    # plt.show()

    # task 3
    origin, ilpf_pic, fs_pic, ilpf_fs_pic = pic.ideal_lowpass_filter(R=30.0)
    paste(1, origin, "origin", model=plt.cm.gray)
    paste(2, ilpf_pic, "ideal_lowpass_filter", model=plt.cm.gray)
    paste(3, np.log(np.abs(fs_pic)), "fftshift_frequency_pic", model=plt.cm.gray)
    paste(4, np.log(np.abs(ilpf_fs_pic)), "fs_ilpf_frequency_pic", model=plt.cm.gray)
    plt.show()
    

if __name__ == "__main__":
    main()


'''
在网上寻找一张福州大学校园图像，并将之转换为灰度图像，完成以下题目：
1编程实现陷波滤波器，对该图进行频率域滤波。
2编程实现巴特沃思低通滤波器，对该图进行图像滤波。
3编程实现理想低通滤波器，对该图进行图像滤波，并分析一下振铃现象。
将程序代码和实验结果图上传。编程语言不限。
'''
