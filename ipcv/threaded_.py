import numpy as np
import cv2
import time
import threading
import logging

import ipcv
from threading import Thread

from multiprocessing import Pool

# dst+=src*kernelF[np.arange(len(kernelF))]
# w, h = np.meshgrid(np.arange(1,hight-1), np.arange(1,width-1))
# print(h.shape)
# print(w.shape)

import queue


def uni(arr, src):
    freq = {}
    for val in arr:
        if val not in freq:
            if val == -1:
                freq[val] = np.negative(src)
            elif val == 1:
                freq[val] = src

            else:
                freq[val] = np.dot(src, val)

    return freq


def filter2D(src, dstDepth, kernel, delta=0, maxCount=255):
    PAD_H = int(((kernel.shape[0] - 1) // 2)) * 2
    PAD_W = int((kernel.shape[1] - 1) // 2) * 2

    w = src.shape[1] // 2
    h = src.shape[0] // 2

    src_imgs = []
    for x in range(0, src.shape[1], w):
        for y in range(0, src.shape[0], h):
            src_imgs.append(src[x:x + w + PAD_W, y:y + h + PAD_H])

    threads = [None] * len(src_imgs)
    results = [None] * len(src_imgs)

    for i in range(len(threads)):
        threads[i] = Thread(target=filter2D_threaded, args=(src_imgs[i], dstDepth, kernel, delta, maxCount, results, i))
        threads[i].start()

    for i in range(len(threads)):
        threads[i].join()

    img1 = np.concatenate((results[0], results[2]), axis=0)
    img2 = np.concatenate((results[1], results[3]), axis=0)
    img = np.concatenate((img1, img2), axis=1)

    return img


def filter2D_threaded(src, dstDepth, kernel, delta, maxCount, results, i):
    # startTime = time.time()

    kernelF = kernel.flatten()

    PAD_H = int((kernel.shape[0] - 1) / 2)
    PAD_W = int((kernel.shape[1] - 1) / 2)

    h =  src.shape[0] - 2*PAD_H
    w =  src.shape[1] - 2*PAD_W

    frequencies = uni(kernelF, src)


    if (len(src.shape) > 2):
        dst = np.zeros(shape=(h, w, 3))
    else:
        dst = np.zeros(shape=(h, w))


    for y in range(-PAD_H, PAD_H + 1):
        n1 = np.arange(PAD_H + y, src.shape[0] - PAD_H + y)  # up down (left right )
        for x in range(-PAD_W, PAD_W + 1):
            n2 = np.arange(PAD_W + x, src.shape[1] - PAD_W + x)  # col (left right )
            # dst += frequencies[kernel[y + PAD_H, x + PAD_W]][n1, :][:, n2]
            dst+= frequencies[kernel[y + PAD_H, x + PAD_W]][PAD_H+y:src.shape[0]+y-1,PAD_W+x:src.shape[1]+x-1]

    dst /= len(kernelF)
    results[i] = ((dst + delta).astype(dstDepth))
    # print('thread:{} Elapsed time = {} [s]\n'.format(i,(time.time() - startTime)))
    return results
