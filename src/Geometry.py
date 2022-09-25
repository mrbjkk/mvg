import numpy as np
import cv2

class Geometry(object):
    def __init__():
        pass
    
    def homogenization(self, x:np.ndarray, homo_factor=1):
        assert isinstance(type(x), np.ndarray), 'Incorrect input type, ndarray is expected'
        if homo_factor == 0:
            return np.expand_dims(x)

    def zeroVector(self, dims, axis=0):
        if axis == 0:
            return np.zeros((1, dims))
        elif axis == 1:
            return np.zeros((dims, 1))
        