
'''

author aaron berghash

DFT 1D and 2D


         N-1
ft[b] = Sigma X[n]* e^(-j*2*pi*b*n)
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




does COls of img then take the result of the dft on the cols and does it on the rows
'''
import numpy as np


def dft(f, scale=True):
    # setting data type
    f = np.asarray(f, dtype=np.complex128)
    # getting the length of the vector
    length = int(f.shape[0])

    n = np.arange(length) # each index number in 1D signal 0 -> len(f)

    k = n.reshape((length, 1)) # reshape so that mult will be a 2d mat
    const = (-2j * np.pi)

    # scaled or not
    if scale:
        return (1/length)*np.dot(np.exp(const * k * n / length), f) # sigma over all indexs in the signal
    else:
        return np.dot(np.exp(const * k * n / length), f) # sigma over all indexs in the signal

# does dft on cols first then on the new img rows

def dft2(f, scale=True):
    return dft(dft(f,scale).T,scale).T



if __name__ == '__main__':
    import numpy
    import time

    M = 2 ** 5
    N = 2 ** 5
    f = numpy.ones((M, N), dtype=numpy.complex128)
    print()
    repeats = 10
    print('Repetitions = {0}'.format(repeats))

    startTime = time.time()
    for repeat in range(repeats):
        mine = dft2(f,False)
    string = 'Average time per transform = {0:.8f} [s] '
    string += '({1}x{2}-point DFT2)'
    print(string.format((time.time() - startTime) / repeats, M, N))

    startTime = time.time()
    for repeat in range(repeats):
        F = numpy.fft.fft2(f)
    string = 'Average time per transform = {0:.8f} [s] '
    string += '({1}x{2}-point FFT2)'
    print(string.format((time.time() - startTime) / repeats, M, N))

    print("--------------- compare ---------------")
    print("Numpy fft2 == my dft2: ", numpy.allclose(F, mine))
    print("---------------------------------------")