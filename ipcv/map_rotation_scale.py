'''
author aaron berghash amb8489
'''

import numpy as np
import cv2
import ipcv
import os.path
import time


'''
    essientally what this function does 
    
    for y in raneg (m):
     for x in raneg (n):
       a.append([y-MAPy//2,x-maxx//2, 1])
       
       
    makes a grid
    


'''

def indices_array_generic_translated(r, c):


    r0 = np.arange(r) - (r // 2)
    r1 = np.arange(c) - (c // 2)
    out = np.empty((r, c, 3), dtype=int)
    out[:, :, 0] = r0[:, None]
    out[:, :, 1] = r1
    out[:, :, 2] = 1
    return out.reshape(r * c, 3, 1)



'''
always row major order for scalers and translate 

'''

def map_rotation_scale(src, rotation=0, scale=[1, 1], translate=[0, 0]):
    # degrees to radians
    rotation = np.radians(rotation)

    # find new map size based off of src img:
    imgSize = src.shape

    # scale translation
    Sy = scale[0]
    Sx = scale[1]

    # not in funcyion but these translate
    Ty = translate[0]
    Tx = translate[1]

    ySize = imgSize[0]
    xSize = imgSize[1]

    # topLEFT , topRIGHT , bottomLEFT ,  bottomRIGHT

    cornners = [np.matrix([[0], [0], [1]]), np.matrix([[0], [(xSize)], [1]]), np.matrix([[(ySize)], [0], [1]]),
                np.matrix([[(ySize)], [(xSize)], [1]])]

    # -------------MAKE ROTATON TRANSFIRM SCALE MATRIX -------------

    # rotaion mat counter clockwise
    R = np.mat([[np.cos(rotation), np.sin(rotation), 0],
                [-np.sin(rotation), np.cos(rotation), 0],
                [0, 0, 1]])

    # scale an transform mat

    ST = np.mat([[1 / Sy, 0, Tx / Sx],
                 [0, 1 / Sx, -Ty / Sy],
                 [0, 0, 1]])

    # Rotation scale transform mat
    RST = np.matmul(R, ST)

    # -------------find new map size by rotation 4 corners  -------------

    # mult all values by rst

    cornners = np.matmul(R, cornners)

    # find size my difference of the max and min x cord and differnce of the max and min y

    # max min x
    newWIDTHmax = max(cornners[0, 1] * Sx, cornners[1, 1] * Sx, cornners[2, 1] * Sx, cornners[3, 1] * Sx)
    newWIDTHmin = min(cornners[0, 1] * Sx, cornners[1, 1] * Sx, cornners[2, 1] * Sx, cornners[3, 1] * Sx)

    # max min y
    newHEIGHTmax = max(cornners[0, 0] * Sy, cornners[1, 0] * Sy, cornners[2, 0] * Sy, cornners[3, 0] * Sy)
    newHEIGHTmin = min(cornners[0, 0] * Sy, cornners[1, 0] * Sy, cornners[2, 0] * Sy, cornners[3, 0] * Sy)

    newHeight = int(newHEIGHTmax - newHEIGHTmin)
    newWidth = int(newWIDTHmax - newWIDTHmin)

    # -------------MAKE MAP -------------

    # make ALphine map a matrix of index cords / TRANSLATE CORDS to rotae in center of new size

    AlfineIDXMap = (indices_array_generic_translated(newHeight, newWidth)) 
    # ROTATE scale trans

    XYPRIMES = np.asarray(np.matmul(RST, AlfineIDXMap))

    # TRANSLATE CORDS BACK
    XYPRIMES[:, 0] += ySize // 2
    XYPRIMES[:, 1] += xSize // 2

    # Y primes , X primes
    return XYPRIMES[:, 0].reshape(newHeight, newWidth), XYPRIMES[:, 1].reshape(newHeight, newWidth)
