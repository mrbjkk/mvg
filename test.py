import cv2
import numpy as np
from solution import PlanarGeometry

if __name__ == '__main__':
    # pg = PlanarGeometry()
    # # point = np.ones((2, 1))
    # # point = np.array([[1],[1]])
    # point = np.array([1, 0])
    # t = np.array([0, 0])
    theta = 90
    # image_path = 'data/000000000285.jpg'
    # # img = utils.image_read(image_path, True)
    # ret = pg.isometric_trans2d(point, theta, t, 1)
    # # ret = pg.point_lies_on_theline(x1, line1)
    # # ret = pg.intersection_of_lines(line1, line2)
    # # print(ret)

    trans_matrix3D = np.mat(
        [
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    c = np.linalg.eig(trans_matrix3D)
    print('hello')
