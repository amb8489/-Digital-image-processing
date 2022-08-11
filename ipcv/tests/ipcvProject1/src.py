
## author Aaron Berghash amb8489
import numpy as np
import cv2
import time
import ipcv

## applies a LUT to a img as long as the LUT is 1D

def applyLookUpTable(img, lut):
    return cv2.LUT(img, lut).astype(np.uint8)

## reurns if the img has more than 1 channel

def isColorImg(img):
    return len(img.shape) > 2

## makes a histogram of img
## FOR COLOR  the RGB channels are summed together to make one 1D hist so that rgb ratios are the same after enhanced
def mkImgHistogram(img):
    cols = len(img[0])
    rows = len(img)
    hist = np.zeros((256))

    if isColorImg(img):
        nChanels = img.shape[2]
        for y in range(0, rows):
            for x in range(0, cols):
                for c in range(0, nChanels):
                    hist[img[y, x, c]] += 1
        hist /= nChanels
    else:
        for y in range(0, rows):
            for x in range(0, cols):
                hist[img[y, x]] += 1
    return hist

# makes simple prob density function using a hist (h) and an img

def mkPDF(h, img):
    return h / (img.shape[0] * img.shape[1])


### makes simple CDF using given PDF

def mkCDF(pdf):
    return np.cumsum(pdf)


## makes a LUT for equlize by multiplying the cdf by max count

def mkLUTEqu(cdf, maxcount):
    return cdf * maxcount


## does histgtam transformation from
## imgDC -> imgProbability -> min(abs(targetprobsMAT - imgProbability)) -> minvalue of that matrixs' -> that vals index
##

def mkLUTEmatch(cdfImg, cdfTartget):
    EImg = np.zeros((cdfImg.shape))

    for i in range(0, 256):
        idx = (np.abs(cdfTartget - cdfImg[i])).argmin()
        EImg[i] = idx

    return np.around(EImg)


#simple % trimmed of img hist.. mapped line to dc out
def trimmed(TrimmedPercent, im, maxCount=255):
    dcMax = np.max(im)
    dcMin = np.min(im)

    trimmedAmount = (dcMax - dcMin) * TrimmedPercent
    dcMinTrimmed = round(dcMin + trimmedAmount)
    dcMaxTrimmed = round(dcMax - trimmedAmount)
    slope = maxCount / (dcMaxTrimmed - dcMinTrimmed)
    b = maxCount - (slope * dcMaxTrimmed)
    lut = np.ndarray(maxCount + 1)

    for i in range(0, 256):
        DCout = (slope * i) + b
        if i > dcMaxTrimmed:
            DCout = maxCount
        if i < dcMinTrimmed:
            DCout = 0
        lut[i] = int(DCout)
    return applyLookUpTable(im, lut)

def histogram_enhancement(im, etype='linear2', target=None, maxCount=255):
    histIN = mkImgHistogram(im)
    pdfIN = mkPDF(histIN, im)
    cdfIN = mkCDF(pdfIN)

    if etype == 'equalize':
        imgLUTEqu = mkLUTEqu(cdfIN, maxCount)
        outIMG = applyLookUpTable(im, imgLUTEqu)

    if etype == 'match':
        if len(target.shape) > 1:
            histMatch = mkImgHistogram(target)
            pdfMatch = mkPDF(histMatch, target)
            cdfMatch = mkCDF(pdfMatch)
        else:
            cdfMatch = mkCDF(target)

        elut = mkLUTEmatch(cdfIN, cdfMatch)
        outIMG = applyLookUpTable(im, elut)

    ## parsing etype for the word linear and then taking the value after the 'r' casting it as a float and passing that into
    ## general % trimmed
    if etype[:6] == 'linear':
        print(float(etype.split("r")[1]) / 100)
        outIMG = trimmed(float(etype.split("r")[1]) / 100, im, maxCount)


    #
    # if etype == 'linear2':
    #     outIMG = trimmed(.02, im, maxCount)
    # if etype == 'linear1':
    #     outIMG = trimmed(.01, im, maxCount)

    return outIMG

if __name__ == '__main__':
    import cv2
    import ipcv
    import os.path
    import time
    import numpy as np



    # nums = [1,2,3,4,5,6,7,8,9]
    #
    # for i in range(1,len(nums)):
    #     nums[i]+=nums[i-1]
    # print(nums)

    home = os.path.expanduser('~')
    filename = home + os.path.sep + 'src/python/examples/data/giza.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/redhat.ppm'
    filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
    filename = home + os.path.sep + 'src/python/examples/data/giza.jpg'


    matchFilename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
    matchFilename = home + os.path.sep + 'src/python/examples/data/redhat.ppm'
    matchFilename = home + os.path.sep + 'src/python/examples/data/giza.jpg'
    matchFilename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
    matchFilename = home + os.path.sep + 'src/python/examples/data/redhat.ppm'




    im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    print('Filename = {0}'.format(filename))
    print('Data type = {0}'.format(type(im)))
    print('Image shape = {0}'.format(im.shape))
    print('Image size = {0}'.format(im.size))

    cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename, im)
    ipcv.flush()

    print('Linear 2% ...')
    startTime = time.time()
    enhancedImage = histogram_enhancement(im, etype='linear2')
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))
    cv2.namedWindow(filename + ' (Linear 2%)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Linear 2%)', enhancedImage)
    ipcv.flush()

    print('Linear 1% ...')
    startTime = time.time()
    enhancedImage = histogram_enhancement(im, etype='linear1')
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))
    cv2.namedWindow(filename + ' (Linear 1%)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Linear 1%)', enhancedImage)
    ipcv.flush()

    print('Equalized ...')
    startTime = time.time()
    enhancedImage = histogram_enhancement(im, etype='equalize')
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))
    cv2.namedWindow(filename + ' (Equalized)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Equalized)', enhancedImage)
    ipcv.flush()

    tgtIm = cv2.imread(matchFilename, cv2.IMREAD_UNCHANGED)
    print('Matched (Image) ...')
    startTime = time.time()
    enhancedImage = histogram_enhancement(im, etype='match', target=tgtIm)
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))
    cv2.namedWindow(filename + ' (Matched - Image)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Matched - Image)', enhancedImage)
    ipcv.flush()

    tgtPDF = np.ones(256) / 256
    print('Matched (Distribution) ...')
    startTime = time.time()
    enhancedImage = histogram_enhancement(im, etype='match', target=tgtPDF)
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))
    cv2.namedWindow(filename + ' (Matched - Distribution)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Matched - Distribution)', enhancedImage)

    action = ipcv.flush()
