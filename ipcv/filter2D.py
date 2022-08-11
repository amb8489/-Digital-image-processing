'''
by aaron berghsh


a mulit threaded version or img kernal

breaks img up into 4 blocks and each block is computed and then all 4 threads concat their blocks together


'''

import numpy as np
import cv2
import time
import ipcv
from threading import Thread

'''
this function is ment to reduce the number of times the src arr is to be multiplyed

for each unique number in the kernal it will map the number to the src array multiplyed by that number 


src = [1,2,3]
freq[2] = [2,4,6]

really where the apllying of the img kernal happens 
'''


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


'''

makes 4 threads and breaks img into 4 blocks to be proceesses by a thread then they are all put back together
'''


def filter2D(src, dstDepth, kernel, delta=0, maxCount=255):
    PAD_H = int(((kernel.shape[0] - 1) // 2)) * 2
    PAD_W = int((kernel.shape[1] - 1) // 2) * 2

    w = src.shape[1] // 2
    h = src.shape[0] // 2

    # making blocks

    src_imgs = []
    for x in range(0, src.shape[1], w):
        for y in range(0, src.shape[0], h):
            src_imgs.append(src[x:x + w + PAD_W, y:y + h + PAD_H])

    threads = [None] * len(src_imgs)
    
    results = [None] * len(src_imgs)

    #  making threads with the blocks in them / pproccessing the blocks

    for i in range(len(threads)):
        threads[i] = Thread(target=filter2D_threaded, args=(src_imgs[i], dstDepth, kernel, delta, maxCount, results, i))
        threads[i].start()

    # joing all thr threads so they dont get ahead of eachother

    for i in range(len(threads)):
        threads[i].join()

    # rebuilding blocks
    return np.concatenate(
        (np.concatenate((results[0], results[2]), axis=0), np.concatenate((results[1], results[3]), axis=0)), axis=1)


'''
applys a kermal to an img 

results is a shared resource between threads and i is the index that thread should place its block into results


'''


def filter2D_threaded(src, dstDepth, kernel, delta, maxCount, results, i):
    # startTime = time.time()

    kernelF = kernel.flatten()

    PAD_H = int((kernel.shape[0] - 1) / 2)
    PAD_W = int((kernel.shape[1] - 1) / 2)

    h = src.shape[0] - 2 * PAD_H
    w = src.shape[1] - 2 * PAD_W

    # getting unique number
    uniSrc = uni(kernelF, src)

    # building dst
    if (len(src.shape) > 2):
        dst = np.zeros(shape=(h, w, 3))
    else:
        dst = np.zeros(shape=(h, w))

    # summing dst planes (9 of 'em)
    #  really just summing the approprate section of the src img from the uni function , see uni function
    for y in range(-PAD_H, PAD_H + 1):
        n1 = np.arange(PAD_H + y, src.shape[0] - PAD_H + y)
        for x in range(-PAD_W, PAD_W + 1):
            n2 = np.arange(PAD_W + x, src.shape[1] - PAD_W + x)
            dst += uniSrc[kernel[y + PAD_H, x + PAD_W]][PAD_H + y:src.shape[0] + y - PAD_H,
                   PAD_W + x:src.shape[1] + x - PAD_W]

    results[i] = (np.divide(dst, len(kernelF)) + delta).astype(dstDepth)
    # print('thread:{} Elapsed time = {} [s]\n'.format(i,(time.time() - startTime)))
    return results
