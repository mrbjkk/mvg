import cv2
import numpy as np

# import utils


class Geometry:
    def __init__(self, homo_factor=1):
        self.homo_factor = homo_factor

    def _homogenization(self, x, homo_factor):
        return

    def zeroVector(self, dims, axis=0):
        if axis == 0:
            return np.zeros((1, dims))
        elif axis == 1:
            return np.zeros((dims, 1))


class PlanarGeometry(Geometry):
    def point_lies_on_theline(self, x: np.ndarray, line: np.ndarray):
        x = x.transpose()
        assert x.shape[0] == 3 and x.shape == line.shape, "incorrect input shape"
        return False if np.dot(x, line) else True

    def intersection_of_lines(self, line1, line2):
        line1 = line1.transpose()
        line2 = line2.transpose()
        return np.cross(line1, line2)

    def line_through_points(self, x1, x2):
        return np.cross(x1, x2)

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
    def __init__(self):
        pass

    def _homogenization(self, x, homo_factor=1):
        if homo_factor == 0:
            return np.array([x[0], x[1], 0]).reshape((3, 1))
        else:
            return np.array(
                [x[0] / homo_factor, x[1] / homo_factor, homo_factor]
            ).reshape((3, 1))

    def directLinearTrans(self, x1, x2, homo_factor1=1, homo_factor2=1):
        x1 = [self._homogenization(x, homo_factor1) for x in x1]
        x2 = [self._homogenization(x, homo_factor2) for x in x2]
        # x2 = H x1
        mat_list = [
            np.mat(
                [
                    np.c_[
                        self.zeroVector(3),
                        -x2[i][2, 0] * x1[i].transpose(),
                        x2[i][1, 0] * x1[i].transpose(),
                    ].squeeze(),
                    np.c_[
                        x2[i][2, 0] * x1[i].transpose(),
                        self.zeroVector(3),
                        -x2[i][0, 0] * x1[i].transpose(),
                    ].squeeze(),
                    np.c_[
                        -x2[i][1, 0] * x1[i].transpose(),
                        x2[i][0, 0] * x1[i].transpose(),
                        self.zeroVector(3),
                    ].squeeze(),
                ]
            )
            for i in range(4)
        ]
        linearly_inde_mat_list = [
            np.mat(
                [
                    np.c_[
                        self.zeroVector(3),
                        -x2[i][2, 0] * x1[i].transpose(),
                        x2[i][1, 0] * x1[i].transpose(),
                    ].squeeze(),
                    np.c_[
                        x2[i][2, 0] * x1[i].transpose(),
                        self.zeroVector(3),
                        -x2[i][0, 0] * x1[i].transpose(),
                    ].squeeze(),
                ]
            )
            for i in range(4)
        ]

        mat = np.concatenate(mat_list)
        linearly_inde_mat = np.concatenate(linearly_inde_mat_list)
        u, s, vh = np.linalg.svd(linearly_inde_mat)
        # return vh[:, -1].reshape((3, 3))
        v = vh.transpose()

        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape((9, 1))

        return v[:, -1].reshape(3, 3)

    def _centroid(self, points):
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

    @staticmethod
    def _func1(x, *args):
        dist = 0
        for point in args[0]:
            dist += np.sqrt((x * point[0]) ** 2 + (x * point[1]) ** 2)
        return dist / len(args[0]) - np.sqrt(2)

    def _isotropic_scaling(self, points):
        n = len(points)
        # translate
        t = self._centroid(points)
        # scale
        from scipy.optimize import fsolve

        scale = fsolve(self._func1, 1, args=points)[0]
        # create matrix
        trans_mat = np.array([[scale, 0, -t[0]], [0, scale, -t[1]], [0, 0, 1]])
        return trans_mat

    def normalized_DLT(self, points1, points2, homo_factor1=1, homo_factor2=1):
        # 计算相似变换
        T1 = self._isotropic_scaling(points1)
        T2 = self._isotropic_scaling(points2)
        # 齐次化
        points1 = [self._homogenization(point1) for point1 in points1]
        points2 = [self._homogenization(point2) for point2 in points2]
        # 归一化
        points1_t = [np.dot(T1, point1) for point1 in points1]
        points2_t = [np.dot(T1, point1) for point1 in points1]
        dist = 0
        for x in points1_t:
            dist += np.linalg.norm(x[:2])
        dist = dist / len(points1_t)
        print('hello')
        H_e = self.directLinearTrans(points1_t, points2_t, homo_factor1, homo_factor2)
        H = np.dot(np.dot(np.linalg.inv(T2), H_e), T1)
        return H

    class Cost_Function(Geometry):
        def _homogenization(self, x, homo_factor=1):
            if homo_factor == 0:
                return np.array([x[0], x[1], 0]).reshape((3, 1))
            else:
                return np.array(
                    [x[0] / homo_factor, x[1] / homo_factor, homo_factor]
                ).reshape((3, 1))

        def algebraic_dist(self, x1, x2):
            x1 = self._homogenization(x1).transpose().reshape((3,))
            x2 = self._homogenization(x2).transpose().reshape((3,))
            a = np.cross(x1, x2)
            return a[0] ** 2 + a[1] ** 2

        def geometric_dist(self, x, x_m, x_b, x_p, H, single_image=False):
            """param description

            Support single point or point set
            Keyword arguments:
            x -- measured image coordinates of the first image
            x_m -- estimated values of the points.
            x_b -- true values of the points
            x_p -- measured image coordinates of the second image
            H -- transformation matrix
            Return: geomtric distance
            """

            # error in one image
            if single_image:
                dist = np.linalg.norm(x_p, np.dot(H, x_b))
                return dist
            else:
                dist1 = np.linalg.norm(x, np.dot(np.linalg.inv(H), x_p))
                dist2 = np.linalg.norm(x_p, np.dot(H, x))
                return dist1 + dist2

        def reprojection_error(self, x1, x1_e, x2, x2_e, H_e):
            x2_e = np.dot(H_e, x1_e)

            dist1 = np.linalg.norm(x1, x1_e)
            dist2 = np.linalg.norm(x2, x2_e)
            return dist1 + dist2
