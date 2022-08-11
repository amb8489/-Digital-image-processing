# aaron berghash amb8489



import numpy as np
from matplotlib import pyplot as plt
import math
import cv2

# @param im : img in gray only
# @param maxCount : max display val
# @param verbose : bool weather or not to plot
# @return threshold im and thresh hold


def otsu_threshold(im, maxCount=255, verbose=False):
    # GRAY IMGS ONLY FOR THIS PROJECT FOR PYTHON REQUIRED
    if len(im.shape) < 3:
        # finding the min and max to know where first non and last non zero are in hist
        minDC = int(np.min(im))+1
        maxDC = int(np.max(im))+1

        # img histogram transposed for x axis on botton and total on y axis
        # used opencv hist because we made a hist manually on last project and theirs is much faster
        hist = cv2.calcHist([im], [0], None, [256], [0, 256]).T[0]

        #calculate pdf from the first non zero to the last non zero  (pi)
        pdf = (hist / (im.shape[0] * im.shape[1]))[minDC:maxDC]
        pdfRev = np.flip(pdf)

        #prefix sum cumulitive density
        cdf = np.cumsum(pdf)

        # w(k)  aka cdf
        w0 = np.cumsum(pdf)


        #  for each DC 0 to 255: (mu0[DC] +=  (( pdf[DC]*DC)+ mu0[DC-1] )/ cd[dc])
        mu0 = np.cumsum((np.arange(minDC , maxDC))*pdf)/w0


        # 1 - cdf OR just the reverce of the cumsum pdf
        w1 = np.flip(np.cumsum(pdfRev))

        # ut - u(k) / w1
        mu2 = (np.flip(np.cumsum(pdfRev*((np.flip(np.arange(minDC , maxDC)))))))/w1

        #max idx of sigma^2 array off by min dc-1 because of shortened histogram -1
        thresh = (w1*w0*pow(mu0-mu2,2)).argmax()+minDC

        #vectorized thresholding    diving by thresh makes # less than one if below thesh floored to 0 or > 1 floored to 1 clapped to be 1 if thresh is low then
        # multed by max
        im = np.clip(np.floor(im / thresh),0,1)*maxCount


        # plot if wanted
        if verbose:
            plt.plot(hist,label='histogram of src img')
            plt.axvline(thresh , 0, 1, label='k* = {}'.format(thresh),c='r')
            plt.legend()
            plt.show()
        return im,thresh
    else:
        print("input was a color img, GRAY SCALE ONLY FOR PYTHON ")
        exit(1)
