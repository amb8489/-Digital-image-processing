import numpy as np
import cv2
import time


def uni(arr, src):
    freq = {}
    for val in arr:
        if val not in freq:
            if val == -1:
                print("hi")
                freq[val] = -1*src
            elif val == 1:
                freq[val] = src

            else:
                freq[val] = np.dot(src, val)

    return freq


def filter2D(src, dstDepth, kernel, delta=0, maxCount=255):
    # startTime = time.time()

    kernelF = kernel.flatten()

    PAD_H = int((kernel.shape[0] - 1) / 2)
    PAD_W = int((kernel.shape[1] - 1) / 2)

    h = src.shape[0] - 2 * PAD_H
    w = src.shape[1] - 2 * PAD_W

    frequencies = uni(kernelF, src)

    if (len(src.shape) > 2):
        dst = np.zeros(shape=(h, w, 3))
    else:
        dst = np.zeros(shape=(h, w))
    startTime = time.time()
    print(src)
    print()
    print()

    for y in range(-PAD_H, PAD_H + 1):
        n1 = np.arange(PAD_H + y, src.shape[0] - PAD_H + y)  # up down (left right )
        for x in range(-PAD_W, PAD_W + 1):
            n2 = np.arange(PAD_W + x, src.shape[1] - PAD_W + x)  # col (left right )
            print("\n",frequencies[kernel[y + PAD_H, x + PAD_W]][PAD_H+y:src.shape[0]+y-1,PAD_W+x:src.shape[1]+x-1],"\n"
            "--------------------------------------------------")
    print('---Elapsed time = {0} [s]'.format(time.time() - startTime))

    dst /= len(kernelF)
    return ((dst + delta).astype(dstDepth))
