import numpy as np
import cv2
import time
import ipcv

def applyLookUpTable(img, lut):
    return cv2.LUT(img, lut).astype(np.uint8)

def isColorImg(img):
    return len(img.shape) > 2

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


def mkPDF(h, img):
    return h / (img.shape[0] * img.shape[1])

def mkCDF(pdf):
    return np.cumsum(pdf)

    #for i in range (1,len())
    # pdf[i]+=pdf[i-1]

def mkLUTEqu(cdf, maxcount):
    return cdf * maxcount

def mkLUTEmatch(cdfImg, cdfTartget):
    EImg = np.zeros((cdfImg.shape))

    for i in range(0, 256):
        idx = (np.abs(cdfTartget - cdfImg[i])).argmin()
        EImg[i] = idx

    return np.around(EImg)

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

    if etype[:6] == 'linear':
        outIMG = trimmed(float(etype.split("r")[1]) / 100, im, maxCount)

    # if etype == 'linear2':
    #     outIMG = trimmed(.02, im, maxCount)
    # if etype == 'linear1':
    #     outIMG = trimmed(.01, im, maxCount)

    return outIMG
