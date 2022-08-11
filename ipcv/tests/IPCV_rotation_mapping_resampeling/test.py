# author aaron berghash

import ipcv
import numpy as np
import cv2
import ipcv
import os.path
import time
import math


def Nearest_Neighbor_Interp(src, map1, map2, borderMode, borderValue):
    # INT RETURNED ROUND
    map1 = np.floor((map1 + .5)).astype(int)
    map2 = np.floor((map2 + .5)).astype(int)
    dst = np.ndarray((map1.shape[0], map1.shape[1], 3))

    dstHeight = dst.shape[0]
    dstWidth = dst.shape[1]

    srcHeight = src.shape[0]
    srcWidth = src.shape[1]

    for y in range(dstHeight):
        for x in range(dstWidth):

            YP = map1[y, x]
            XP = map2[y, x]

            if (0 <= XP < srcWidth) and (0 <= YP < srcHeight):

                dst[y, x] = src[YP, XP]
            else:

                if borderMode == ipcv.BORDER_REPLICATE:
                    if XP >= srcWidth and 0 <= YP < srcHeight:

                        dst[y, x] = src[YP, srcWidth - 1]

                    elif XP < 0 and 0 <= YP < srcHeight:
                        dst[y, x] = src[YP, 0]

                    elif YP >= srcHeight and 0 <= XP < srcWidth:

                        dst[y, x] = src[srcHeight - 1, XP]

                    elif YP < 0 and 0 <= XP < srcHeight:
                        dst[y, x] = src[0, XP]
                else:
                    dst[y, x] = borderValue

    return dst.astype(np.uint8)


def Bi_Linear_Interp(src, mapY, mapX, borderMode, borderValue):
    dst = np.ndarray((map1.shape[0], map1.shape[1], 3))
    # dst = np.ndarray((map1.shape[0], map1.shape[1]))


    dstHeight = dst.shape[0]
    dstWidth = dst.shape[1]

    srcHeight = src.shape[0]
    srcWidth = src.shape[1]

    # getting X primes and Y primes that fall within the bounds of the src img getting their idx in map y and x
    e = (np.logical_and(np.logical_and(mapY >= 0, mapY < srcHeight), np.logical_and(mapX >= 0, mapX < srcWidth)))

    x_y_in_DST = np.where(e)
    YP_in_DST = np.extract(e, mapY)
    XP_in_DST = np.extract(e, mapX)

    #savinng org vales before using them to find find closes pt
    Xorg = XP_in_DST
    Yorg = YP_in_DST

    # finding closest pt
    YP_in_DST = np.floor(YP_in_DST)
    XP_in_DST = np.floor(XP_in_DST)

    YP_in_DST = YP_in_DST
    XP_in_DST = XP_in_DST

    Ys = YP_in_DST
    Xs = XP_in_DST

    Xplusone = (XP_in_DST + 1)
    Yplusone = (YP_in_DST + 1)

    #clipping closest pixel for edge cases for all pixels

    Ys, Yplusone = np.clip([Ys, Yplusone], 0, src.shape[0] - 1).astype(int)
    Xs, Xplusone = np.clip([Xs, Xplusone], 0, src.shape[1] - 1).astype(int)
    Xs = np.asarray(Xs)
    Ys = np.asarray(Ys)

    #getting the values for the closes points  ie ; dc[y,x]  dc[y,x+1] dc[y+1,x]  dc[y+1,x+1] for ALL pixels
    b = np.arange(len(Xs))
    Ia = np.asarray((src[Ys[b], Xs[b]]))
    Ib = np.asarray((src[Yplusone[b], Xs[b]]))
    Ic = np.asarray((src[Ys[b], Xplusone[b]]))
    Id = np.asarray((src[Yplusone[b], Xplusone[b]]))

    #interp for all pixels

    wa = (Xplusone - Xorg) * (Yplusone - Yorg)
    wb = (Xplusone - Xorg) * (Yorg - Ys)
    wc = (Xorg - Xs) * (Yplusone - Yorg)
    wd = (Xorg - Xs) * (Yorg - Ys)

    out = (Ia.T * wa).T + (Ib.T * wb).T + (Ic.T * wc).T + (Id.T * wd).T

    #filling all interpted vals  in map
    a = np.arange(len(x_y_in_DST[0]))
    x_y_in_DST = np.asarray(x_y_in_DST)

    dst[x_y_in_DST[0,a], x_y_in_DST[1,a]] = out[a]

    return dst.astype(np.uint8)


def remap(src, map1, map2, interpolation=ipcv.INTER_NEAREST, borderMode=ipcv.BORDER_CONSTANT, borderValue=0):
    if interpolation == ipcv.INTER_NEAREST:
        return Nearest_Neighbor_Interp(src, map1, map2, borderMode, borderValue)

    elif interpolation == ipcv.INTER_LINEAR:
        return Bi_Linear_Interp(src, map1, map2, borderMode, borderValue)


if __name__ == '__main__':
    import sys

    # np.set_printoptions(threshold=np.inf)
    home = os.path.expanduser('~')
    filename = home + os.path.sep + 'src/python/examples/data/bow.png'
    src = cv2.imread(filename, 1)

    map1, map2 = ipcv.map_rotation_scale(src, 76, [3, 2])

    print("img size in: {}   img size out: {}".format(src.shape, map1.shape))

    # img = remap(src, map1, map2, ipcv.INTER_NEAREST, ipcv.BORDER_CONSTANT, 0)
    # ipcv.show(img)
    # ipcv.flush()

    startTime = time.time()
    img = remap(src, map1, map2, ipcv.INTER_LINEAR, ipcv.BORDER_CONSTANT, 128)
    elapsedTime = time.time() - startTime
    print('Elapsed time (map creation) = {0} [s]'.format(elapsedTime))

    ipcv.show(img)
