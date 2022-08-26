import cv2
import numpy as np
# from src.solution import *
import solution

if __name__ == "__main__":

    # point = pt_2d._homogenization(point)
    x1 = [np.array([1, 1]), np.array([2, 2]), np.array([3, 1]), np.array([4, 2])]
    x1 = [np.array([-1, -1]), np.array([-2, -2]), np.array([-3, -1]), np.array([-4, -2])]
    x2 = [np.array([1, 2]), np.array([2, 3]), np.array([3, 2]), np.array([4, 3])]

    pt_2d = solution.Estimation_2D()
    mat = pt_2d.normalized_DLT(x1, x2)
    print("hello")
