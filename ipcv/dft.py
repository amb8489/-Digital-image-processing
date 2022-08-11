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
'''
import numpy as np


def dft(f, scale=True):
    f = np.asarray(f, dtype=np.complex128)
    # getting the length of the vector
    length = int(f.shape[0])
    n = np.arange(length)
    k = n.reshape((length, 1)) # reshape so that mult will be a 2d mat
    const = (-2j * np.pi)
    if scale:
        return (1/length)*np.dot(np.exp(const * k * n / length), f) # sigma over all indexs in the signal
    else:
        return np.dot(np.exp(const * k * n / length), f) # sigma over all indexs in the signal

if __name__ == '__main__':
    import numpy
    import time
    print()

    N = 2 ** 12
    f = numpy.ones(N, dtype=numpy.complex128)

    repeats = 10
    print('Repetitions = {0}'.format(repeats))

    startTime = time.time()
    for repeat in range(repeats):
        mine = dft(f,False)
    string = 'Average time per transform = {0:.8f} [s] ({1}-point DFT)'
    print(string.format((time.time() - startTime) / repeats, len(f)))

    startTime = time.time()
    for repeat in range(repeats):
        F = numpy.fft.fft(f)
    string = 'Average time per transform = {0:.8f} [s] ({1}-point FFT)'
    print(string.format((time.time() - startTime) / repeats, len(f)))
    print("--------------- compare ---------------")
    print("Numpy fft == my dft: ",numpy.allclose(F,mine))
    print("---------------------------------------")

