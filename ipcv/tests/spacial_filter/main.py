
'''

aaron Berghash
'''
if __name__ == '__main__':
    import cv2
    import os.path
    import time
    import numpy as np
    import ipcv
    from threading import Thread

    home = os.path.expanduser('~')
    filename = home + os.path.sep + 'src/python/examples/data/checkerboard.tif'
    filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
    filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'


    src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    dstDepth = ipcv.IPCV_8U
    # kernel = np.asarray([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    # offset = 128
    kernel = np.asarray([[-1 ,-1 ,-1] ,[-1 ,8 ,-1] ,[-1 ,-1 ,-1]])
    offset = 128
    # kernel = np.ones((15,15))
    # offset = 0
    # kernel = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    # offset = 0


#-------------------------------------------------------------------------------#
    startTime = time.time()
    dst = ipcv.filter2D(src, dstDepth, kernel, delta=offset)
    print('Total Elapsed time = {0} [s]'.format(time.time() - startTime))
    print("Type:" ,dst.dtype)

    cv2.namedWindow(filename + ' (Filtered)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Filtered)', dst)
    action = ipcv.flush()

    cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename, src)
    action = ipcv.flush()
