# author aaron berghash

import ipcv
import numpy as np
import math
import time
import cv2

'''
coverts img to c lab
'''


def toCLAB(img):
    # defaut D6d ref
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)


'''
calcs simil filter 
'''


def simColorFilter(neighborhood, pad):
    return neighborhood - neighborhood[pad, pad]


def distClab(x, y, tx, ty):
    return math.sqrt(pow(tx - x, 2) + pow(ty - y, 2))


'''
calcs dist
'''


def dist(x, y, tx, ty):
    return math.sqrt(pow(tx - x, 2) + pow(ty - y, 2))


'''
gets a chuck of the img the neighboring pixels 
'''


def getNeighbor(src, pad, x, y):
    return src[y - pad: y + pad + 1, x - pad: x + pad + 1].astype(np.int)


'''
calcs a gaussian filter 2d
'''


def gaussian(arr, sig):
    kernel = np.asarray(np.exp(-0.5 * np.square((arr / sig))))
    return kernel


'''
calcs a cloness filter  filter 2d
'''


def closness_filter(diamitor, pad, sigmaDistance):
    kern = np.zeros((diamitor, diamitor))

    for y in range(diamitor):
        for x in range(diamitor):
            kern[y, x] = dist(x, y, pad, pad)

    closeness = gaussian(kern, sigmaDistance)
    return closeness / closeness[pad, pad]


'''
preforms bilatera; filter on img given sigmaDistance, sigmaRange, d=-1
'''


def bilateral_filter__(src, sigmaDistance, sigmaRange, d=-1, borderType=ipcv.BORDER_WRAP, maxCount=255):
    srcHeight = src.shape[0]
    srcWidth = src.shape[1]

    # where d is calculated

    radius = d * sigmaRange
    if d < 0:
        radius = 2 * sigmaDistance

    print("d:", d)
    print("radius:", radius)
    print("sigmaDistance:", sigmaDistance)
    print("sigmaRange:", sigmaRange)

    dst = np.zeros((src.shape))

    diamitor = (2 * radius) + 1
    pad = (diamitor // 2)

    # boarder wrappig / const / replicate

    if borderType == ipcv.BORDER_WRAP:
        src = np.pad(src, ((pad, pad), (pad, pad)), 'wrap')
    if borderType == ipcv.BORDER_CONSTANT:
        src = np.pad(src, ((pad, pad), (pad, pad)), 'constant', constant_values=((128, 128), (128, 128)))
    if borderType == ipcv.BORDER_REPLICATE:
        src = np.pad(src, ((pad, pad), (pad, pad)), 'edge')

    # color img conver to clab space l chan
    colorSim = src
    if isColor(src):
        colorSim = toCLAB(src)[:, :, 0]

    closeness = closness_filter(diamitor, pad, sigmaDistance)
    srcHeight = src.shape[0]
    srcWidth = src.shape[1]

    # np.pad(image_stack, ((extra_left, extra_right), (extra_top, extra_bottom), (0, 0)),
    #        mode='constant', constant_values=3)

    for y in range(pad, srcHeight - pad):
        if y % 50 == 0:
            print("{}% complete".format(round(((y - pad) / (srcHeight - pad)) * 100)))
        for x in range(pad, srcWidth - pad):
            # the color distance filter
            neighborhood = getNeighbor(colorSim, pad, x, y)
            sim = simColorFilter(neighborhood, pad)
            s = gaussian(sim, sigmaRange)

            # the BL filter
            s *= closeness

            # applying the filter
            dst[y - pad, x - pad] = np.sum(np.multiply(neighborhood, s)) / np.sum(s)

    return np.clip(dst.astype(np.uint8), 0, maxCount)


'''
if a img is color orr not 
'''


def isColor(src):
    if len(src.shape) > 2:
        total = 0
        for i in range(20):
            for j in range(20):
                pix = src[i, j]
                r = pix[0]
                g = pix[1]
                b = pix[2]
                if (r == b and b == g):
                    total += 1
                    if total > 20:
                        return False

        return True
    else:
        return False


def bilateral_filter(src, sigmaDistance, sigmaRange, d=-1, borderType=ipcv.BORDER_WRAP, maxCount=255):
    print("running ...")
    if isColor(src):
        print("COLOR IMG")

        dst = np.stack([
            bilateral_filter__(src[:, :, 0], sigmaDistance, sigmaRange, d, borderType, maxCount),

            bilateral_filter__(src[:, :, 1], sigmaDistance, sigmaRange, d, borderType, maxCount),

            bilateral_filter__(src[:, :, 2], sigmaDistance, sigmaRange, d, borderType, maxCount)], axis=2)
        return dst

    else:

        if len(src.shape) > 2:
            print("GRAY IMG")
            dst = bilateral_filter__(src[:, :, 0], sigmaDistance, sigmaRange, d, borderType, maxCount),
            return dst

        else:

            print("GRAY IMG")
            dst = bilateral_filter__(src, sigmaDistance, sigmaRange, d, borderType, maxCount)
            return dst
