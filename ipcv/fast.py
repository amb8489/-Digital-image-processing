'''
author aaron bergash

'''
import numpy as np
import ipcv

'''
gets the 16 circle neighbors given an x y
'''
def get_circle_neighbors(src, x, y):
    return np.asarray(
        [src[y + 3, x - 1],src[y + 3, x], src[y + 3, x + 1], src[y + 2, x + 2], src[y + 1, x + 3], src[y, x + 3], src[y - 1, x + 3],
         src[y - 2, x + 2], src[y - 3, x + 1], src[y - 3, x],
         src[y - 3, x - 1], src[y - 2, x - 2], src[y - 1, x - 3], src[y, x - 3], src[y + 1, x - 3], src[y + 2, x - 2]])


'''
tests for Contiguous strip of 1's

'''

def hadContiguousThreshold(arr,contiguousThreshold):
    if np.sum(arr)<contiguousThreshold:
        return False
    for i in range(0,16):
        a = np.roll(arr, i)
        a = a[:contiguousThreshold]
        if np.all(a == 1 ):
            return True
    return False


'''
FAST corrnner detection implemntation
'''
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

def fast(src, differenceThreshold=50, contiguousThreshold=12, nonMaximalSuppression=True):

    if isColor(src):
        print("Gray scale img only")
        exit(1)



    pad = 3
    srcHeight,srcWidth = src.shape
    dst = np.zeros(src.shape)

    for y in range(pad, srcHeight - pad):
        if y%50 ==0:
            print("still running...")
        for x in range(pad, srcWidth - pad):
            n = get_circle_neighbors(src,x,y)
            diff = np.where(n >= src[y,x]+differenceThreshold, 1, 0)
            if hadContiguousThreshold(diff,contiguousThreshold):

                dst[y,x] = 1
            diff = np.where(n <= src[y,x]-differenceThreshold, 1, 0)
            if hadContiguousThreshold(diff, contiguousThreshold):
                dst[y, x] = 1


    return dst


