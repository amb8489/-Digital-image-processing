'''
author aaron berghash amb8489
'''

import numpy as np
import math
import time
import ipcv
from numpy.linalg import inv

'''
    def columstack(arr1, arr2)
        if len(arr1) == len(arr2)
            e = []
            for i in range (len(srcY)):
                e.append([srcY[i],srcX[i]])
            return e
        :return None



'''


def map_gcp(src, map, srcX, srcY, mapX, mapY, order=2):
    n = ((order + 1) * ((order + 1) + 1)) / 2

    # gets idx grid kinda inefficent to call the entire function rather than just making the grid

    map1, map2 = ipcv.map_rotation_scale(map, 0, [1, 1])


    map1= map1.flatten()
    map2= map2.flatten()

    XY_Primes_src = np.column_stack([srcY, srcX, np.ones(len(srcX))])

    a = []
    b = []

    for y_exp in range(order+1):
        for x_exp in range(order  - y_exp+1):
            print(x_exp, y_exp)
            a.append(np.asarray([(np.power(mapY, y_exp) * np.power(mapX, x_exp))]).T)
            b.append(np.asarray([(np.power(map1, y_exp) * np.power(map2, x_exp))]).T)



    model_mat = np.asarray(np.column_stack(a))
    new_map = np.asarray(np.column_stack(b))

    model_mat_transpose = np.transpose(model_mat)
    model_mat_inverse = np.linalg.inv((np.matmul(model_mat_transpose, model_mat)))
    ivT = np.matmul(model_mat_inverse, model_mat_transpose)




    coeffecients = np.matmul(ivT, XY_Primes_src)

    print("coeffecients x y:\n",coeffecients)
    map_new = np.matmul(coeffecients.T, new_map.T)
    y = map_new[0].reshape(map.shape[0], map.shape[1])
    x = map_new[1].reshape(map.shape[0], map.shape[1])


    return y, x
