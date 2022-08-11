'''

author aaron berghash

DFT 1D and 2D


         N-1
ft[b] = Sigma X[n]* e^(j*2*pi*b*n)
         n:0



 vectorized would be extract the common repetitive computations and do them all SIMD:

 for every idx : muliply and sum aka (dot prod) :

 (every value in the signal) * e^( (const) * (curr sig idx #) * (every index #))
 --------------------------------------------------------------------------


 e^( (const) * (every curr idx ) * (all index))/len n

 (every value in the signal)



 k * n = for each k mult it by all n... [[k=1 : (1 * 1), (1* 2), ... ,(1*n)]:
                                        [k=2 : (2 * 1), (2* 2), ... ,(2*n)]
                                        [k=k : (k * 1), (k* 2), ... ,(k*n)]]


some pseudo code i wrote:

k*n is a 2d mat containing all the for each idx in the signal * current idx in the sumation, this doesnt change and is a
const as well

next: for i in range(0,nRow) : acc = 0 for idx in range(0,nRow): acc+= signal[i] * e^ const * kn[i][idx] | dft[i] = acc

O(n^2) fir each n go sum and mult all n
'''
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



if __name__ == '__main__':
    import numpy
    import time

    M = 2 ** 5
    N = 2 ** 5
    F = numpy.zeros((M, N), dtype=numpy.complex128)
    F[0, 0] = 1
    print()

    repeats = 10
    print('Repetitions = {0}'.format(repeats))

    startTime = time.time()
    for repeat in range(repeats):
        mine = idft2(F)
    string = 'Average time per transform = {0:.8f} [s] '
    string += '({1}x{2}-point iDFT2)'
    print(string.format((time.time() - startTime) / repeats, M, N))

    startTime = time.time()
    for repeat in range(repeats):
        f = numpy.fft.ifft2(F)
    string = 'Average time per transform = {0:.8f} [s] '
    string += '({1}x{2}-point iFFT2)'
    print(string.format((time.time() - startTime) / repeats, M, N))
    print("--------------- compare ---------------")
    print("Numpy ifft2 == my idft2: ",numpy.allclose(f,mine))
    print("---------------------------------------")
