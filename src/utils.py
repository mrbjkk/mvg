import numpy as np
import cv2


def centroid(points):
    """
    x -- a list of points
    """
    n = len(points)
    x, y = 0, 0
    for point in points:
        x += point[0]
        y += point[1]
    return np.array([x / n, y / n])
