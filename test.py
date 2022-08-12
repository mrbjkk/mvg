import cv2
import numpy as np
from src.solution import *

if __name__ == "__main__":
    pt_2d = Estimation_2D()
    point = np.array([2, 2])

    # point = pt_2d._homogenization(point)
    # x1 = [np.array([1, 1]), np.array([2, 2]), np.array([3, 1]), np.array([4, 2])]

    # x2 = [np.array([1, 2]), np.array([2, 3]), np.array([3, 2]), np.array([4, 4])]
    # transH = pt_2d.directLinearTrans(x1, x2)

    # ret = np.dot(transH, point)
    x1 = np.array([1, 0])
    x2 = np.array([0, 1])
    ret = pt_2d.algebraic_dist(x1, x2)
    print("hello")
