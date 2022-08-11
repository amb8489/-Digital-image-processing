
import cv2

from ipcv.constants import *
from ipcv.flush import *

from ipcv.harris import *
from ipcv.fast import *

from ipcv.histogram_enhancement import *
from ipcv.quantize import quantize
from ipcv.otsu_threshold import otsu_threshold

from ipcv.remap import *
from ipcv.map_rotation_scale import *
from ipcv.map_gcp import *


from ipcv.PointsSelected import *

from ipcv.map_quad_to_quad import *

from ipcv.filter2D import *

from ipcv.character_recognition import *
from ipcv.matched_char_recognition import *
from ipcv.angle_char_recognition import *
from ipcv.bilateral_filter_ import *
# binds the cvs imshow and the flush action together
def show(img,name = "img"):
    cv2.namedWindow(name, cv2.WINDOW_GUI_EXPANDED)
    cv2.imshow(name , img)
    action = flush()

def showA(img,name = "img",delay = 100):
    cv2.namedWindow(name, cv2.WINDOW_GUI_EXPANDED)
    cv2.imshow(name , img)
    action = flush_Animate(delay)
