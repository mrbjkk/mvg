import cv2
import numpy as np
from src.solution import *

if __name__ == "__main__":
    pt_2d = Estimation_2D()
    cost_function = pt_2d.Cost_Function()
    point = np.array([2, 2])

    # point = pt_2d._homogenization(point)
    x1 = [np.array([1, 1]), np.array([2, 2]), np.array([3, 1]), np.array([4, 2])]
    x2 = [np.array([1, 2]), np.array([2, 3]), np.array([3, 2]), np.array([4, 4])]
    # x1 = [np.zeros((1, 1)), np.zeros([2, 2]), np.zeros([3, 1]), np.zeros([4, 2])]
    # x2 = [np.zeros((1, 2)), np.zeros([2, 3]), np.zeros([3, 2]), np.zeros([4, 4])]
    # img1 = cv2.imread('data/000000000285.jpg')
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = np.zeros_like(img1)

    # x = img1[x1[0][0], x1[0][1]]

    transH = pt_2d.directLinearTrans(x1, x2)
    ret = pt_2d.apply_trans(transH, np.array([2,2]))

    # ret = np.dot(transH, point)
    # x1 = np.array([1, 0])
    # x2 = np.array([0, 1])
    # ret = cost_function.algebraic_dist(x1=x1, x2=x2)
    print("hello")
