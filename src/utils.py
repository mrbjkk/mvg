import numpy as np
import cv2

from scipy import optimize


def centroid(points):
    """
    x -- a list of points
    return the centroid of a set of 2D points
    """
    n = len(points)
    x, y = 0, 0
    for point in points:
        x += point[0]
        y += point[1]
    return np.array([x / n, y / n])


def linequation_2D(point1, point2):
    sign = 1
    a = point2[1] - point1[1]
    if a < 0:
        sign = -1
        a = sign * a
    b = sign * (point1[0] - point2[0])
    c = sign * (point2[1] * point2[0] - point1[0] * point2[1])
    return [a, b, c]

def line_func(x, A, B):
    return A * x + B

# def linequation_2D(points):
#     sign = 1
#     a = point2[1] - point1[1]
#     if a < 0:
#         sign = -1
#         a = sign * a
#     b = sign * (point1[0] - point2[0])
#     c = sign * (point2[1] * point2[0] - point1[0] * point2[1])
#     return [a, b, c]

def curve_fit_2D(points, func=line_func):
    [x, y] = [
        [points[i][0] for i in range(len(points))],
        [points[j][1] for j in range(len(points))],
    ]
    model = optimize.curve_fit(func, x, y)[0]
    return model

def distbetweenpoint_line(point, line):
    dist = np.abs(line[0] * point[0] + line[1] * point[1] + line[2]) / np.sqrt(
        line[0] ** 2 + line[1] ** 2
    )
    return dist
