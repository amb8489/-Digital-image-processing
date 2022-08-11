


# author aaron berghahs amb8489

import math
import numpy
import cv2

## takes img in and returns if its has 3 channels i.e a color img ir not
def isColorImg(img):
    return len(img.shape)>=3


## look up take for %d to reduce mod computatins
def mkLutModD(d):
    lut = []
    for i in range(0, 256):
        lut.append(i % d)
    return lut


# METHOD DEFINITION
def quantize(img, levels, qtype='uniform', maxCount=255, displayLevels=None):
    # multiplier
    if displayLevels !=None:
        d = math.floor(displayLevels / levels)
    else:
        d = math.floor(maxCount / levels)


    if qtype == 'uniform ' or qtype == 'igs':

        #igs quant
        if qtype == 'igs':
            lutModD = mkLutModD(d)
            cols = len(img[0])
            rows = len(img)

            #color igs
            if isColorImg(img):
                # error diffusion for color imgs
                for y in range(0, rows):
                    for x in range(0, cols):
                        if x == cols - 1:
                            x = -1
                            y = (y + 1) % rows
                        for c in range(0, img.shape[2]):
                            r = lutModD[img[y,x][c]]
                            if img[y,x + 1][c] + r <= 255:
                                img[y,x + 1][c] += r
            else: # non color igs
                for y in range(0, rows):
                    for x in range(0, cols):
                            if x == cols - 1:
                                x = -1
                                y = (y + 1) % rows
                            r = lutModD[img[y,x]]
                            if img[y,x + 1] + r <= 255:
                                img[y,x + 1] += r

    # quantizing uniform
    img = img // d
    img = img * (d)

    return img