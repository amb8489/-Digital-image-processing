# author aaron berghash

import ipcv
import numpy as np
import cv2
import os.path
import math

'''
        how np.where() and np.extract() and bool arrays are used to vectorize
        
        example of getting pixels heigher and lowwer than src height  outside of (0-srcH)
    
        extract, like where gets the indexes same as where given e THEN goes into given array at the index and gets the values at array
        gets Y/X primes in maps y x at the index where e was met
        example :
        src_height = 8
        
         mapy= -1 -1 -1
                0  0  0
                9  9  8
         e = logical_or(mapy<0 || mapy> height) returns a truth table bool array same size as mapy with truth vals at the idx
         e =  T T T
              f f f
              T T F
          idxs_where_true = where(e)
          idxs_where_true --> [[0,0],[0,1],[0,2],[2,0],[2,1]] returns idx of where true in array, e
          a = 0 1 2
              3 4 5
              6 7 8
         extract(e,a) gets indexs( i.e. where(e)): [[0,0],[0,1],[0,2],[2,0],[2,1]] foreach index, RETURNS : [a[index0],a[index1],a[index2] ]
         extract(e,a)>> [0,1,2,6,7]
        
        
        concept used for map1 and map2 to find wanted pixels faster verses nested for loops
                
        
'''


#----------------------------------------------------------------------------------------
'''

:return a img with the specified border mode applied 

'''

def ApplyBorderMode(img, src, srcShape, borderMode, borderValue, mapY, mapX):
    srcHeight = srcShape[0]
    srcWidth = srcShape[1]

    # all pixel locations outside img
    # e is a logical condtion to ceck for in an array -> returns a bool array of condtion in given arays map x,y
    e = (np.logical_or(np.logical_or(mapY <= 0, mapY > srcHeight - 1), np.logical_or(mapX <= 0, mapX > srcWidth - 1)))
    # where gives us the index in the bool array e where condtion e was met in arrays in e
    x_y_in_DST = np.asarray(np.where(e))
    idx = np.arange(len(x_y_in_DST[0]))

    if borderMode == ipcv.BORDER_CONSTANT:
        img[x_y_in_DST[0, idx], x_y_in_DST[1, idx]] = [borderValue, borderValue, borderValue]
    elif borderMode == ipcv.BORDER_REPLICATE:

        # getting x and y primes

        YP = np.asarray(np.extract(e, mapY))
        XP = np.asarray(np.extract(e, mapX))

        # truncating x and ys to fit in respective height and width of src
        YP = np.clip(YP, 0, src.shape[0] - 1).astype(int)
        XP = np.clip(XP, 0, src.shape[1] - 1).astype(int) \
            # mapping
        img[x_y_in_DST[0, idx], x_y_in_DST[1, idx]] = src[YP[idx], XP[idx]]
    else:
        print("borderMode not suported: ".format(borderMode))
        exit(1)

    return img.astype(np.uint8)


'''
preforms nearest neighbor int  on img given src by rounding x y maps 

'''

def Nearest_Neighbor_Interp(src, mapY, mapX):
    # INT RETURNED ROUND
    mapY = np.floor((mapY + .5)).astype(int)
    mapX = np.floor((mapX + .5)).astype(int)

    # maping new img
    dst = np.ndarray((mapY.shape[0], mapY.shape[1], 3))

    # getting some info on the new img and src img
    dstHeight = dst.shape[0]
    dstWidth = dst.shape[1]

    srcHeight = src.shape[0]
    srcWidth = src.shape[1]

    # finding pixels in mapx and map y that are mapped into the src aka pixels for our new img that fall within src
    e = (np.logical_and(np.logical_and(mapY >= 0, mapY < srcHeight), np.logical_and(mapX >= 0, mapX < srcWidth)))
    x_y_in_DST = np.asarray(np.where(e))
    YP_in_DST = np.asarray(np.extract(e, mapY))
    XP_in_DST = np.asarray(np.extract(e, mapX))
    idx = np.arange(len(x_y_in_DST[0]))

    # getting the  dc vales map 1 and map 2 mapped to in src (rounded nearest to ) and seting them in the new img at the new img loc , y x prime
    dst[x_y_in_DST[0, idx], x_y_in_DST[1, idx]] = src[YP_in_DST, XP_in_DST]

    return dst.astype(np.uint8)



#======------------------------------- bilinear interp


'''
def bilinearinterp(src ,x y):
    for each closes pixel:
    P1 src[round(y,x)]
    P2 src[round(y,x)]
    P3 src[round(y,x)]
    P4 src[round(y,x)]
    
    interp horizontaily  p1-p2  and p3 p4
    
    
    
    interp between    A  p1-p2   B
                          |
                        dc x' y'
                          |
                      C  p3 p4   D
    
    :return dc x' y'
    

    


def bi lin interp:

for y in range(dstHeight):
    for x in range(dstHeight):
        yp = mapY[y,x]
        xp = mapX[y,x]
        if (0<= yp < srcHeight  and 0<= xp < width)
            dst[y,x] = bilinearinterp(src,yp,xp)
        else:
            boarder val


'''


def Bi_Linear_Interp(src, mapY, mapX):
    dst = np.ndarray((mapY.shape[0], mapY.shape[1], 3))

    dstHeight = dst.shape[0]
    dstWidth = dst.shape[1]

    srcHeight = src.shape[0]
    srcWidth = src.shape[1]

    # getting X primes and Y primes that fall within the bounds of the src img getting their idx in map y and x
    e = (np.logical_and(np.logical_and(mapY >= 0, mapY < srcHeight), np.logical_and(mapX >= 0, mapX < srcWidth)))

    x_y_in_DST = np.where(e)
    YP_in_DST = np.extract(e, mapY)
    XP_in_DST = np.extract(e, mapX)

    # savinng org vales before using them to find find closes pt
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

    # clipping closest pixel for edge cases for all pixels bc of the xs and ys  + 1

    Ys, Yplusone = np.clip([Ys, Yplusone], 0, src.shape[0] - 1).astype(int)
    Xs, Xplusone = np.clip([Xs, Xplusone], 0, src.shape[1] - 1).astype(int)
    Xs = np.asarray(Xs)
    Ys = np.asarray(Ys)


    # getting the values for the closes points  , dc[y,x]  dc[y,x+1] dc[y+1,x]  dc[y+1,x+1] for ALL pixels
    b = np.arange(len(Xs))
    Ia = np.asarray((src[Ys[b], Xs[b]]))
    Ib = np.asarray((src[Yplusone[b], Xs[b]]))
    Ic = np.asarray((src[Ys[b], Xplusone[b]]))
    Id = np.asarray((src[Yplusone[b], Xplusone[b]]))

    # interp for all pixels :  ex:
    # A = (dc[i+1,j] - dc[i,j]) *(x' - i) +dc[i,j]
    # B = (dc[i+1,j+1] - dc[i,j+1]) *(x' - i) +dc[i,j+1]
    # interp between A -- dc[x',y'] -- B = equation for out below

    wa = (Xplusone - Xorg) * (Yplusone - Yorg)
    wb = (Xplusone - Xorg) * (Yorg - Ys)
    wc = (Xorg - Xs) * (Yplusone - Yorg)
    wd = (Xorg - Xs) * (Yorg - Ys)
    # interp between A -- dc[x',y'] -- B = form for out below for ALL pixels and there closes neighbor
    out = (Ia.T * wa).T + (Ib.T * wb).T + (Ic.T * wc).T + (Id.T * wd).T

    # filling all interpted vals  in map
    a = np.arange(len(x_y_in_DST[0]))
    x_y_in_DST = np.asarray(x_y_in_DST)
    dst[x_y_in_DST[0, a], x_y_in_DST[1, a]] = out[a]

    return dst.astype(np.uint8)



#--------------------------remap

'''
remapes the src img on to new map using map1 and map2 that are x y in new img map to map to src 

'''


def remap(src, map1, map2, interpolation=ipcv.INTER_NEAREST, borderMode=ipcv.BORDER_CONSTANT, borderValue=90):
    if interpolation == ipcv.INTER_NEAREST:
        imgOUT = Nearest_Neighbor_Interp(src, map1, map2)

    elif interpolation == ipcv.INTER_LINEAR:
        imgOUT = Bi_Linear_Interp(src, map1, map2)
    else:
        print("interpolation method not suported: ".format(interpolation))
        exit(1)

    return ApplyBorderMode(imgOUT, src, src.shape, borderMode, borderValue, map1, map2)


# -------------------------------------------TESTS
#
# if __name__ == '__main__':
#     import sys
#     import time
#
#     np.set_printoptions(threshold=np.inf)
#     home = os.path.expanduser('~')
#     filename = home + os.path.sep + 'src/python/examples/data/lenna copy.tif'
#     src = cv2.imread(filename, 1)
#
#     map1, map2 = ipcv.map_rotation_scale(src, 101, [5, 1])
#
#     print("img size in: {}   img size out: {}".format(src.shape, map1.shape))
#     startTime = time.time()
#
#     img = remap(src, map1, map2, ipcv.INTER_NEAREST, ipcv.BORDER_CONSTANT, 128)
#     elapsedTime = time.time() - startTime
#     print('Elapsed time (nearest) = {0} [s]'.format(elapsedTime))
#     ipcv.show(img)
#
#     startTime = time.time()
#     img = remap(src, map1, map2, ipcv.INTER_LINEAR, ipcv.BORDER_REPLICATE, 128)
#     elapsedTime = time.time() - startTime
#     print('Elapsed time (bi-linear) = {0} [s]'.format(elapsedTime))
#
#     ipcv.show(img)
