'''

aaron berghash amb8489
'''
import numpy as np
from ipcv import *
from threading import *
import cv2

'''
builds index map
'''
def indices_array_generic(r, c):
    r0 = np.arange(r)
    r1 = np.arange(c)
    out = np.empty((r, c, 3), dtype=int)
    out[:, :, 0] = r0[:, None]
    out[:, :, 1] = r1
    out[:, :, 2] = 1
    return out.reshape(r * c, 3, 1)


'''
img img to be mapped 
map img for src to be mapped on to 

img X/y the four cornneers of src img

mapx/y where in the map the img should be transformed to 


returms x y map to be remapeped on to
'''

def map_quad_to_quad(img, map, imgX, imgY, mapX, mapY):


    imgHeight = map.shape[0]
    imgWidth = map.shape[1]
    # --------------------------------------------------------b-------------------------------------------------#

    srcMat = np.asarray([imgY[0:3],imgX[0:3],[1,1,1]])
    srcWeights =np.matmul(np.linalg.inv(srcMat),np.asarray([[imgY[3]],[imgX[3]],[1]]))

    b = np.array(
    [[srcWeights[0,0]*srcMat[0,0],srcWeights[1,0]*srcMat[0,1],srcWeights[2,0]*srcMat[0,2] ],
    [srcWeights[0, 0] * srcMat[1, 0], srcWeights[1, 0] * srcMat[1, 1], srcWeights[2, 0] * srcMat[1, 2]],
    [ srcWeights[0,0],            srcWeights[1,0],            srcWeights[2,0]]])

    # ---------------------------------------------------------a------------------------------------------------#

    mapMat = np.asarray([mapY[0:3],mapX[0:3],[1,1,1]])

    mapWeights =np.matmul(np.linalg.inv(mapMat),np.asarray([[mapY[3]],[mapX[3]],[1]]))

    a = np.array(
    [[mapWeights[0,0]*mapMat[0,0],mapWeights[1,0]*mapMat[0,1],mapWeights[2,0]*mapMat[0,2]],
    [ mapWeights[0,0]*mapMat[1,0],mapWeights[1,0]*mapMat[1,1],mapWeights[2,0]*mapMat[1,2]],
    [ mapWeights[0,0],            mapWeights[1,0],            mapWeights[2,0]]]            )

    # ------------------------------------------------------projection MAT------------------------------------------------#

    projMat = np.matmul(b,np.linalg.inv(a))

   #----------------- basically Alfine but diff mat from last proj -------------

    # preforming transform
    proIDXMap = (indices_array_generic(imgHeight, imgWidth))
    XY = np.asarray(np.matmul(projMat, proIDXMap))

    # dehomoginizing
    XY = XY / np.array(XY[:,2])[:, None]

    # print("-----------------------------srcmat-------------------------------\n",srcMat)
    # print("-----------------------------srcWeights-------------------------------\n",srcWeights)
    # print("-----------------------------B-------------------------------\n",b)
    # print("-----------------------------mapMat-------------------------------\n",mapMat)
    # print("-----------------------------mapWeights-------------------------------\n",mapWeights)
    # print("-----------------------------A-------------------------------\n",a)
    # print("-----------------------------proj-------------------------------\n",projMat)

    return np.float32(XY[:, 1].reshape(imgHeight,imgWidth)),np.float32( XY[:, 0].reshape(imgHeight,imgWidth))




