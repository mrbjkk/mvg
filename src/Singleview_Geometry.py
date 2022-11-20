import scipy
import numpy as np
import cv2


from Geometry import Geometry


class Camera_Model(Geometry):
    def __init__(
            self,
            focal_len: float,
            center_offset: np.ndarray,
            rotate_angle: np.ndarray,
            camera_center_world_coord: np.ndarray,
            pix_factor=np.array([1, 1]),
            distorb=0,
    ):
        self.focal_len = focal_len
        self.rotate_angle = rotate_angle
        self.center_offset = center_offset * pix_factor
        self.camera_center_world_coord = camera_center_world_coord
        self.distorb = distorb

    def _basic_pinhole_model(self):
        focal_len = self.focal_len
        diag = [focal_len, focal_len, 1]
        mat = np.hstack(np.diag(diag), self.zeroVector(3, 1))
        return mat

    def _calibration_mat(self):
        focal_len = self.focal_len
        camera_offset = self.center_offset
        distorb = self.distorb
        mat = np.mat([
            [focal_len, distorb, camera_offset[0]],
            [0, focal_len, camera_offset[1]],
            [0, 0, 1],
        ])
        return mat

    def _rotation_mat(self):
        rotate_angle = self.rotate_angle
        roll, pitch, yaw = rotate_angle[0], rotate_angle[1], rotate_angle[2]
        rot_mat_roll = np.mat([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1],
        ])
        rot_mat_pitch = np.mat([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ])
        rot_mat_yaw = np.mat([
            [1, 0, 0],
            [0, np.cos(yaw), -np.sin(yaw)],
            [0, np.sin(yaw), np.cos(yaw)],
        ])
        rotation_mat = np.dot(np.dot(rot_mat_roll, rot_mat_pitch), rot_mat_yaw)
        return rotation_mat

    def world2camera(self):
        camera_center = self.camera_center_world_coord
        R = self._rotation_mat()
        upper = np.hstack((R, -1 * np.dot(R, camera_center)))
        lower = np.hstack((self.zeroVector(3), 1))
        mat = np.vstack((upper, lower))
        return mat

    def camera_mat(self):
        camera_center = self.camera_center_world_coord
        translate: np.ndarray = -1 * np.dot(self._rotation_mat(),
                                            camera_center)
        rt_mat = np.hstack((self._rotation_mat(), translate.T))
        camera_mat = np.dot(self._calibration_mat(), rt_mat)
        return camera_mat

    def depth(self, world_point):
        camera_center = self.camera_center_world_coord
        M = self.camera_mat()[:, :3]
        T = 1
        m3_t = M[2]
        w = np.dot(m3_t, (world_point - camera_center))
        sign_det_M = np.sign(np.linalg.det(M))
        depth = sign_det_M * w / T / np.linalg.norm(m3_t.T)
        return depth


class Camera_Despose(Geometry):
    def __init__(self, camera_mat):
        assert isinstance(camera_mat, np.ndarray) or isinstance(
            camera_mat, np.matrix
        ), f'incorrect camera_mat type: f{type(camera_mat)}, np.matrix or np.ndarray is expected'
        self.camera_mat = camera_mat

    def get_camera_center(self):
        camera_mat = self.camera_mat
        p = []
        for col in range(camera_mat.shape[1]):
            p.append(camera_mat[:,col].reshape((3,1)))
        
        x = np.linalg.det(np.concatenate((p[1], p[2], p[3]), axis=1))
        y = -1 * np.linalg.det(np.concatenate((p[0], p[2], p[3]), axis=1))
        z = np.linalg.det(np.concatenate((p[0], p[1], p[3]), axis=1))
        t = -1 * np.linalg.det(np.concatenate((p[0], p[1], p[2]), axis=1))
        return self.vec1D_transpose(np.array([x, y, z, t]))

    def get_matrix(self):
        camera_mat = self.camera_mat
        M = camera_mat[:, :3]
        q, r = np.linalg.qr(M)
        return r, q

# class Affine_Camera(Camera_Model):
#     def __init__():