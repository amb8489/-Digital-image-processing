import numpy as np


def invert_255_and_0(v1):
    return abs(v1 - 255)


def closeness(v1, v2):

    return np.sum(v1*v2)*len(np.where(v2!=0)[0])


def normalize(v1):
    v1 = np.divide(v1, sum(v1.flatten()))
    return v1


import ipcv
import cv2


def run(characterImages):
    bl = characterImages[18]

    mask = characterImages[19]

    bl = bl.astype(np.uint8)

    ipcv.show(cv2.addWeighted(bl.astype(np.uint8), .5, mask.astype(np.uint8), .2, 0.0), "mask")

    correctness = closeness(normalize(invert_255_and_0(bl)), normalize(invert_255_and_0(mask)))

    print(correctness)
