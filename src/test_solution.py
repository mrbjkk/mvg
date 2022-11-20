import cv2
import numpy as np
from pyparsing import FollowedBy

# from src.solution import *
import solution
from Singleview_Geometry import Camera_Model, Camera_Despose

if __name__ == "__main__":

    # point = pt_2d._homogenization(point)
    x1 = [
        np.array([1, 1]),
        np.array([2, 2]),
        np.array([3, 1]),
        np.array([4, 2])
    ]
    # x1 = [np.array([-1, -1]), np.array([-2, -2]), np.array([-3, -1]), np.array([-4, -2])]
    x2 = [
        np.array([1, 2]),
        np.array([2, 3]),
        np.array([3, 2]),
        np.array([4, 3])
    ]

    ransac_set = [
        np.array([1, 1]),
        np.array([2, 2]),
        np.array([3, 1]),
        np.array([4, 2]),
        np.array([1, 2]),
        np.array([2, 3]),
        np.array([3, 2]),
        np.array([4, 3]),
    ]
    # pt_2d = solution.Estimation_2D()
    # mat = pt_2d.normalized_DLT(x1, x2)
    # ret = pt_2d.RANSAC_2D(ransac_set, 2)

    # image_path = "data/input"
    # pt_2d.estimate_homography_2D(
    #     "data/input/stereo_pair1.jpg", "data/input/stereo_pair2.jpg"
    # )

    focal_len = 0.5
    # center_offset = np.array([0.5, 0.5])
    # rotate_angle = np.array([1, 2, 3])
    # camera_center = np.array([0.1, 0.2, 0.3])
    # camera_model = Camera_Model(focal_len, center_offset, rotate_angle, camera_center)
    # camera_mat = camera_model.camera_mat()
    # ret = camera_model.depth(world_point=np.array([0.5, 0.5, 0.5]))
    camera_mat = np.array([[3.53553e2, 3.39645e2, 2.7744e2, -1.44946e6],
                           [-1.03528e2, 2.33212e2, 4.59607e2, -6.32525e5],
                           [7.07107e-1, -3.53553e-1, 6.12372e-1, -9.18559e2]])
    camera_model = Camera_Despose(camera_mat)
    # ret = camera_model.get_camera_center()
    ret = camera_model.get_matrix()
    pass
