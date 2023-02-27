import numpy as np
import cv2


class Geometry(object):
    def __init__():
        pass

    def homogenization(self, x: np.ndarray, homo_factor=1):
        assert isinstance(
            type(x), np.ndarray
        ), "Incorrect input type, ndarray is expected"
        if homo_factor == 0:
            return np.expand_dims(x)

    def zeroVector(self, dims, axis=0):
        if axis == 0:
            return np.zeros((1, dims))
        elif axis == 1:
            return np.zeros((dims, 1))

    def vec1D_transpose(self, vector: np.ndarray):
        return vector.reshape((vector.shape[0], 1))


class AnalyticGeometry(Geometry):
    def centroid(points):
        """ get the centroid of a set of points
        x -- a list of points
        return the centroid of a set of 2D points
        """
        n = len(points)
        x, y = 0, 0
        for point in points:
            x += point[0]
            y += point[1]
        return np.array([x / n, y / n])