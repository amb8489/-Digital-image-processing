
'''


aaron berghash




happy holidays and have a good break  :)


'''



##############funtions to animate the proccess'#########################################
def flush_Animate(delay):
   k = cv2.waitKey(delay)
   action = 'continue'

   return action
def showA(img,name = "img",delay = 100):
    cv2.namedWindow(name, cv2.WINDOW_GUI_EXPANDED)
    cv2.imshow(name , img)
    action = flush_Animate(delay)
########################inverse to see img from its fft##########################################

def ifft(f_spectrum):
    freq_filt_img = np.fft.ifft2(f_spectrum)
    freq_filt_img = np.abs(freq_filt_img)
    return freq_filt_img


if __name__ == '__main__':
    import cv2
    import os.path
    import numpy as np

    home = os.path.expanduser('~')
    filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
    src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    height, width = src.shape



# fft

    f = np.fft.fft2(src)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = fshift


# log mag fft
    magnitude = np.log(np.abs(fshift))
    magnitude *= magnitude[height // 2, width // 2]

# set up
    sumIMG = np.zeros((height, width), np.complex128)
    feq = np.zeros((height, width),np.uint8)
    magnitude1 = np.array(magnitude)

    delay = 1
    for i in range(0,width*height):
        showA(src,"Original Image",delay)

        showA(magnitude1.astype(np.uint8), "Fourier Transform - log(magnitude)", delay)

        y,x = np.unravel_index(np.argmax(magnitude, axis=None), magnitude.shape)
        feq[y,x] = magnitude[y,x]
        magnitude[y,x] = -1
        showA(feq,"Fourier Coefficients Used - log(magnitude)",delay)




        sumIMG[y,x] = magnitude_spectrum[y,x]
        sum = ifft(sumIMG)
        showA(sum.astype(np.uint8),"Summed Components",delay)



        xs = np.linspace(-2 * np.pi, 2 * np.pi, width)
        ys = np.linspace(-2 * np.pi, 2 * np.pi, height)
        dx, dy = np.meshgrid(xs, ys)

        angle = np.angle(magnitude_spectrum[y,x])
        amp = np.sin((angle * dx) + dy)

        showA(amp,"Current Component (Scaled)",delay)




