import numpy as np
import cv2
import time


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
  print(src)
  print()
  a = [1,2,3]

  print(src.flatten()*9)



if __name__ == '__main__':
    import cv2
    import os.path
    import time
    import numpy as np
    import ipcv
    import threaded_

    home = os.path.expanduser('~')
    filename = home + os.path.sep + 'src/python/examples/data/redhat.ppm'
    filename = home + os.path.sep + 'src/python/examples/data/checkerboard.tif'
    filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'
    filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'
    filename = home + os.path.sep + 'Desktop/tp.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
    filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
    filename = home + os.path.sep + 'Desktop/tp.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'
    filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
    filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'
    filename = home + os.path.sep + 'Desktop/tp.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
    filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'
    filename = home + os.path.sep + 'Desktop/tp.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
    filename = home + os.path.sep + 'Desktop/tp.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
    # filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'
    # filename = home + os.path.sep + 'Desktop/tp.jpg'
    # filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'
    filename = home + os.path.sep + 'Desktop/tp.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'
    filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'
    filename = home + os.path.sep + 'src/python/examples/data/redhat.ppm'
    filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'
    filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
    filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'
    filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'
    filename = home + os.path.sep + 'src/python/examples/data/checkerboard.tif'

    filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'
    filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'
    filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'

    filename = home + os.path.sep + 'src/python/examples/data/redhat.ppm'
    filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
    filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'


























    src = np.arange(100).reshape(10,10)

    # src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)


    dstDepth = ipcv.IPCV_8U
    # kernel = np.asarray([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    # offset = 0
    kernel = np.asarray([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    offset = 128
    # kernel = np.ones((15,15))
    # offset = 0
    # kernel = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    # offset = 0

    startTime = time.time()
    dst = filter2D(src, dstDepth, kernel, delta=offset)

    # [:, n2])  # middle ie -1 0 , 0 0 , 1 0 and concat left then right