import numpy as np
import cv2


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


def distbetweenpoint_line(point, line):
    dist = np.abs(line[0] * point[0] + line[1] * point[1] + line[2]) / np.sqrt(
        line[0] ** 2 + line[1] ** 2
    )
    return dist
