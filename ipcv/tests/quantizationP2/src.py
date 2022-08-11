











if __name__ == '__main__':
    import cv2
    import os.path
    import numpy as np
    import ipcv

    home = os.path.expanduser('~')
    filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'
    src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    levels = 8

    d = 256//levels
    blocksize = 10


    src = (np.floor(src/d)*d).astype(np.uint8)
    ipcv.show(src)

    for y in range(0,src.shape[0],blocksize):
        for x in range(0, src.shape[1], blocksize):
             src[y:y + blocksize, x:x + blocksize] = [np.average(src[y:y + blocksize,x:x + blocksize,0]),np.average(src[y:y + blocksize,x:x + blocksize,1]),np.average(src[y:y + blocksize,x:x + blocksize,2])]
    ipcv.show(src)