import cv2
import numpy as np


class Geometry:
    def __init__(self):
        return

    def homogenization(self, x, homo_factor):
        return

    def zeroVector(self, dims, axis=0):
        if axis == 0:
            return np.zeros((1, dims))
        elif axis == 1:
            return np.zeros((dims, 1))


class PlanarGeometry(object):
    def point_lies_on_theline(self, x, line):
        x = x.transpose()
        assert x.shape[0] == 3 and x.shape == line.shape, 'incorrect input shape'
        return False if np.dot(x, line) else True

    def intersection_of_lines(self, line1, line2):
        line1 = line1.transpose()
        line2 = line2.transpose()
        return np.kron(line1, line2)

    def line_through_points(self, x1, x2):
        return np.kron(x1, x2)

    def line_tan2C(self, point, conic):
        """二次曲线过x的切线"""
        return np.dot(conic, point)

    def isometric_trans2d(self, point, theta, t, eps=1, homo_factor=1):
        """isometric transform in 2d plane

        Keyword arguments:
        point -- single 2d point or point set
        theta -- rotation angle
        t     -- translation vector
        eps   -- 1 or -1

        Return: transformed single point or point set
        """

        # assert point.shape[0] == 2 or point, 'incorrect point shape'
        theta = theta * np.pi / 180
        isom_matrix = np.mat(
            [
                [eps * np.cos(theta), -np.sin(theta), t[0]],
                [eps * np.sin(theta), np.cos(theta), t[1]],
                [0, 0, 1],
            ]
        )
        # homogenization for single point
        point = np.array(
            [point[0] * homo_factor, point[1] * homo_factor, homo_factor]
        ).reshape((3, 1))

        ret = np.dot(isom_matrix, point)
        return ret

    def similarity_trans2d(self, point, theta, t, s, homo_factor=1):
        theta = theta * np.pi / 180
        simi_matrix = np.mat(
            [
                [s * np.cos(theta), -s * np.sin(theta), t[0]],
                [s * np.sin(theta), s * np.cos(theta), t[2]],
                [0, 0, 1],
            ]
        )

        point = np.array(
            [point[0] * homo_factor, point[1] * homo_factor, homo_factor]
        ).reshape((3, 1))
        ret = np.dot(simi_matrix, point)
        return ret


class Estimation_2D(Geometry):
    def homogenization(self, x, homo_factor=1):
        if homo_factor == 0:
            return np.array([x[0], x[1], 0]).reshape((3, 1))
        else:
            return np.array(
                [x[0] / homo_factor, x[1] / homo_factor, homo_factor]
            ).reshape((3, 1))

    def directLinearTrans(self, x1, x2, homo_factor1=1, homo_factor2=1):
        x1 = [self.homogenization(x, homo_factor1) for x in x1]
        x2 = [self.homogenization(x, homo_factor2) for x in x2]
        # x2 = H x1
        mat = [
            np.mat(
                [
                    np.c_[
                        self.zeroVector(3),
                        -x2[i][2,0] * x1[i].transpose(),
                        x2[i][1,0] * x1[i].transpose(),
                    ].squeeze(),
                    np.c_[
                        x2[i][2,0] * x1[i].transpose(),
                        self.zeroVector(3),
                        -x2[i][0,0] * x1[i].transpose(),
                    ].squeeze(),
                    np.c_[
                        -x2[i][1,0] * x1[i].transpose(),
                        x2[i][0,0] * x1[i].transpose(),
                        self.zeroVector(3),
                    ].squeeze(),
                ]
            )
            for i in range(4)
        ]
        print('hello')
