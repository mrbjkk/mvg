import numpy as np
import cv2

from Geometry import Geometry

class Camera_Model(Geometry):
    def __init__(
        self,
        camera_center: np.ndarray,
        focal_len: float,
        center_bias: np.ndarray,
        pix_factor=np.array([1, 1]),
    ):
        self.camera_center = camera_center * pix_factor
        self.focal_len = focal_len
        self.center_bias = center_bias * pix_factor

    def _homogenization(self, )
    def _calibration_mat(self):
        focal_len = self.focal_len
        camera_center = self.camera_center
        mat = np.mat(
            [
                [focal_len, 0, camera_center[0]],
                [0, focal_len, camera_center[1]],
                [0, 0, 1],
            ]
        )
