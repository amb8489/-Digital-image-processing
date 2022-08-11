'''

aaron berghash Author
'''

import numpy as np
import ipcv
import cv2
'''
is color img or not 

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



'''
harris corrrner dectection

'''
def harris(src, sigma=1, k=0.04):

    if isColor(src):
        print("Gray scale img only")
        exit(1)


    dy, dx = np.gradient(src)

    # defineing rst of the values of heshen mat
    A = np.square(dx)
    B = np.square(dy)
    C = np.multiply(dx, dy)

    A = cv2.GaussianBlur(A, (0, 0), sigma)
    B = cv2.GaussianBlur(B, (0, 0), sigma)
    C = cv2.GaussianBlur(C, (0, 0), sigma)

    M = [[A, C],
         [C, B]]

    kTrTr = k * np.square((A + B))

    DET = np.multiply(A, B) - np.square(C)
    R = DET - kTrTr

    # Non-Max-Surpression

    pad = 1
    R = np.pad(R, ((pad, pad), (pad, pad)), 'constant', constant_values=((0, 0), (0, 0)))
    srcHeight = R.shape[0]
    srcWidth = R.shape[1]
    corrneers = np.where(R > 0)

    for i in range(len(corrneers[0])):
        y = corrneers[0][i]
        x = corrneers[1][i]
        neighborHood = R[y - pad: y + pad + 1, x - pad: x + pad + 1]
        R[y - pad: y + pad + 1, x - pad: x + pad + 1] = np.where(neighborHood < np.max(neighborHood), 0, neighborHood)
    R = R[pad: srcHeight - pad, pad: srcWidth - pad]

    return R.astype(np.float32)
