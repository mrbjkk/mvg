import cv2
import numpy as np
from src.solution import *

if __name__ == '__main__':
    pt_3d = Estimation_2D()
    x1 = [np.array([1,1]), np.array([1,2]), np.array([1,3]), np.array([1,4])]

    x2 = [np.array([2,1]), np.array([2,2]), np.array([2,3]), np.array([2,4])]
    ret = pt_3d.directLinearTrans(x1, x2)
    print('hello')
