
from ipcv import *

def character_recognition(src, templates,codes,threshold, filterType="angle"):
    if filterType == "angle":
        text, histogram =  ipcv.character_recognition_ang(src, templates,codes,threshold)
        return text, histogram
    else:
        return ipcv.character_recognition_mat(src, templates,codes,threshold)
