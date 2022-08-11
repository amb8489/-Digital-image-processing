import numpy as np


def idft(f, scale=True):
    f = np.asarray(f, dtype=np.complex128)
    length = int(f.shape[0]) # len of signal

    n = np.arange(length) # each index number in 1D signal 0 -> len(f)

    k = n.reshape((length, 1)) # each index number in 1D signal flipped so that result when mult by n is 2D

    const = (2j * np.pi) # const explained above

    # scale or not

    if scale:
        return (1 / length) * np.dot(np.exp(const * k * n / length), f)  # sigma over all indexs in the signal
    else:
        return np.dot(np.exp(const * k * n / length), f)  # sigma over all indexs in the signal



# does dft on cols first then on the new img rows
def idft2(f, scale=True):
    return idft(idft(f,scale).T,scale).T




def idft(f, scale=True):
    f = np.asarray(f, dtype=np.complex128)
    length = int(f.shape[0])  # len of signal

    n = np.arange(length)  # each index number in 1D signal 0 -> len(f)

    k = n.reshape((length, 1))  # each index number in 1D signal flipped so that result when mult by n is 2D

    const = (2j * np.pi)  # const explained above

    # scale or not

    if scale:
        return (1 / length) * np.dot(np.exp(const * k * n / length), f)  # sigma over all indexs in the signal
    else:
        return np.dot(np.exp(const * k * n / length), f)  # sigma over all indexs in the signal


def dft(f, scale=True):
    # setting data type
    f = np.asarray(f, dtype=np.complex128)
    # getting the length of the vector
    length = int(f.shape[0])

    n = np.arange(length)  # each index number in 1D signal 0 -> len(f)

    k = n.reshape((length, 1))  # reshape so that mult will be a 2d mat
    const = (-2j * np.pi)

    # scaled or not
    if scale:
        return (1 / length) * np.dot(np.exp(const * k * n / length), f)  # sigma over all indexs in the signal
    else:
        return np.dot(np.exp(const * k * n / length), f)  # sigma over all indexs in the signal


def dft2shiftMag(f, scale=True):
    ft = np.fft.fftshift(dft(dft(f, scale).T, scale).T)

    return ft, np.absolute(ft)



