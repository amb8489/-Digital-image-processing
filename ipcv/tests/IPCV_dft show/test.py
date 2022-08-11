import numpy as np
import matplotlib as plt
import ipcv
'''
X,Y]=meshgrid(-2:1/16:2,-2:1/16:2); f=sin(2*pi*X+3*pi*Y);
'''

hW, hH = 600, 300
hFreq = 17.5
vFreq = 7.5



# Mesh on the square [0,1)x[0,1)
x = np.linspace( 0, 2*hW/(2*hW +1), 2*hW+1)     # columns (Width)
y = np.linspace( 0, 2*hH/(2*hH +1), 2*hH+1)     # rows (Height)

[X,Y] = np.meshgrid(x,y)
A = np.sin(hFreq*2*np.pi*X + vFreq*2*np.pi*Y)
ipcv.show(A)