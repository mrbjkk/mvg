import os
import numpy as np
import cv2

from scipy import optimize
from tqdm import tqdm


class ImageProc:
    def image_read(self, image_path):
        image_list = []
        for file in os.listdir(image_path):
            img = cv2.imread(image_path + "/" + file)
            image_list.append(img)
        return image_list

    def harris_detector(self,
                        img,
                        gray=False,
                        draw_path=None,
                        resize=0,
                        filt_thres=0.01):
        if resize:
            rows, cols, _channels = map(int, img.shape)
            # img = cv2.pyrDown(img, dstsize=(cols//resize, rows//resize))
            img = cv2.resize(img, dsize=(cols // resize, rows // resize))

        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst = cv2.cornerHarris(img, 2, 3, 0.04)
        if draw_path:
            img[dst > filt_thres * dst.max()] = [0, 0, 255]
            cv2.imwrite(draw_path + 'harris_detector.jpg', img)
        corner_tuple = np.where(dst > filt_thres * dst.max())
        corner_coords = []
        for i in range(len(corner_tuple[0])):
            corner_coords.append((corner_tuple[0][i], corner_tuple[1][i]))
        return corner_coords


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


def line_func_2D(x, A, B):
    # 截距式
    return A * x + B


def linequation_2D(point1, point2):
    sign = 1
    a = point2[1] - point1[1]
    if a < 0:
        sign = -1
        a = sign * a
    b = sign * (point1[0] - point2[0])
    c = sign * (point2[1] * point2[0] - point1[0] * point2[1])
    return [a, b, c]


def curve_fit_2D(sample_points, func=line_func_2D):
    [x, y] = [
        [sample_points[i][0] for i in range(len(sample_points))],
        [sample_points[j][1] for j in range(len(sample_points))],
    ]
    # 得到截距式
    model = optimize.curve_fit(func, x, y)[0].tolist()
    # 转为参数式
    return [model[0], -1, model[1]]


def dist_btw_point_line(point, line):
    dist = np.abs(line[0] * point[0] + line[1] * point[1] +
                  line[2]) / np.sqrt(line[0]**2 + line[1]**2)
    return dist


def show_matching(target_img, reference_img, point_pairs):
    # 拼接
    splicing_img = cv2.hconcat([target_img, reference_img])
    for point_pair in point_pairs:
        cv2.line(
            splicing_img,
            tuple(point_pair[0]),
            tuple([point_pair[1][0] + target_img.shape[0], point_pair[1][1]]),
            color=(0, 255, 0),
        )
    cv2.imwrite('data/output/matching.jpg', splicing_img)
