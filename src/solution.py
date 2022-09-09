from itertools import product
import os
import cv2
import numpy as np

from random import sample
from scipy import optimize

import utils

MAX_NUM = 1000000


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

    def direct_linear_trans(self, x1, x2, homo_factor1=1, homo_factor2=1):
        if x1[0].shape == 2:
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

    @staticmethod
    def _func1(x, *args):
        # nonlinear equation to calculate scale of similar transformation
        dist = 0
        points = args[0]
        centroid = args[1]
        for point in points:
            dist += np.sqrt(
                (x * point[0] - centroid[0]) ** 2 + (x * point[1] - centroid[1]) ** 2
            )
        return dist / len(args[0]) - np.sqrt(2)

    def _isotropic_scaling(self, points):
        # 齐次
        points = [self._homogenization(point) for point in points]
        # translate
        t = utils.centroid(points)

        # scale, fsolve to solve the nonlinear equation
        from scipy.optimize import fsolve

        scale = fsolve(self._func1, 0, args=(points, t))[0]
        # create matrix
        trans_mat = np.array([[scale, 0, -t[0][0]], [0, scale, -t[1][0]], [0, 0, 1]])
        return trans_mat

    def normalized_DLT(self, points1, points2, homo_factor1=1, homo_factor2=1):
        # 计算相似变换
        T1, T2 = self._isotropic_scaling(points1), self._isotropic_scaling(points2)
        # 齐次化
        points1, points2 = [self._homogenization(point1) for point1 in points1], [
            self._homogenization(point2) for point2 in points2
        ]
        # 归一化
        points1_t = [np.dot(T1, point1) for point1 in points1]
        points2_t = [np.dot(T2, point2) for point2 in points2]
        # 计算估计单应
        H_e = self.direct_linear_trans(points1_t, points2_t, homo_factor1, homo_factor2)
        # 解除归一化
        H = np.dot(np.dot(np.linalg.inv(T2), H_e), T1)
        return H

    def _adaptive_num_sampling(self, set_, dist_thres=2, num_sampling=2):
        sample_thres = MAX_NUM
        p = 0.99
        num_set = len(set_)
        sample_count = 0
        while sample_thres > sample_count:
            num_inner_point = 0
            sampler = sample(set_, num_sampling)
            model = utils.curve_fit_2D(sampler)
            for s in set_:
                flag = np.zeros(len(sampler))
                for j in range(len(sampler)):
                    if not (s == sampler[j]).all():
                        flag[j] = 1
                if flag.all():
                    dist = utils.dist_btw_point_line(s, model)
                    if dist < dist_thres:
                        num_inner_point += 1
            epsilon = 1 - num_inner_point / num_set
            sample_thres = np.log(1 - p) // np.log(
                1 - np.power((1 - epsilon), num_sampling)
            )
            sample_count += 1

        size_thres = (1 - epsilon) * num_set
        return sample_thres, size_thres

    def RANSAC_2D(self, set_, num_sampling=2, dist_thres=2):
        """RANSAR_2D

        Args:
            set_ (list): a set of points to be modeled
            num_sampling (int): the number of sample point each time, set to 2 for 2D RANSAC
            dist_thres (int, optional): The distance between other points and the initial model. Defaults to 2.
            size_thres (int, optional): _description_. Defaults to 5.
            num_outlier (int, optional): _description_. Defaults to 2.
            p (float, optional): _description_. Defaults to 0.99.
        """

        sample_thres, size_thres = self._adaptive_num_sampling(
            set_, dist_thres=dist_thres, num_sampling=num_sampling
        )

        ret_list = []
        for i in range(int(sample_thres)):
            inner_point = []
            # 从数据集set中随机选择num_sampling个数据点组成一个样本，并加入内点集
            sampler = sample(set_, num_sampling)
            inner_point += sampler
            # 样本模型，若num_sampling为2，则表示两点确定的一条直线
            model = utils.curve_fit_2D(sampler)
            # 遍历数据集，计算和sampler不同的数据点距离模型(直线)的距离
            for s in set_:
                # 设置标记
                flag = np.zeros(len(sampler))
                for j in range(len(sampler)):
                    if not (s == sampler[j]).all():
                        flag[j] = 1
                if flag.all():
                    dist = utils.dist_btw_point_line(s, model)
                    if dist < dist_thres:
                        # 距离小于阈值dist_thres的点，构成一致集
                        inner_point.append(s)

            if len(inner_point) > size_thres:
                new_model = utils.curve_fit_2D(inner_point)
                ret_list.append(inner_point)

        ret_point = max(ret_list, key=len)
        return new_model, ret_point

    def ssd_matching(
        self,
        target_img,
        reference_img,
        corner_detector='harris',
        resize=0,
        block_size=(10, 10),
        search_block_size=(10, 10),
    ):
        if resize:
            target_img = cv2.resize(target_img, None, fx=resize, fy=resize)
            reference_img = cv2.resize(reference_img, None, fx=resize, fy=resize)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
        target_rows, target_cols = target_img.shape
        reference_rows, reference_cols = reference_img.shape
        if corner_detector == 'harris':
            target_corner = utils.ImageProc.harris_detector(self, target_img)
            reference_corner = utils.ImageProc.harris_detector(self, reference_img)
        best_matching_target, best_matching_reference = [], []
        for coord in target_corner:
            ssd = MAX_NUM
            # for k, j in range(-search_size // 2, search_size // 2):
            target_block = target_img[
                max(0, coord[0] - block_size[0] // 2) : min(
                    coord[0] + block_size[0] // 2, target_rows
                ),
                max(0, coord[1] - block_size[1] // 2) : min(
                    coord[1] + block_size[1] // 2, target_cols
                ),
            ]
            for i, j in product(
                range(-search_block_size[0], search_block_size[0]),
                range(-search_block_size[1], search_block_size[1]),
            ):
                x, y = coord[0] + i, coord[1] + j
                reference_block = reference_img[
                    max(0, x - block_size[0] // 2) : min(
                        max(0, x + block_size[0] // 2), target_rows
                    ),
                    max(0, y - block_size[1] // 2) : min(
                        max(0, y + block_size[1] // 2), target_cols
                    ),
                ]
                if reference_block.shape == target_block.shape:
                    temp_ssd = np.sum((target_block-reference_block)**2)
                    if temp_ssd < ssd:
                        ssd = temp_ssd
                        matching = (coord, [x, y])

            best_matching_target.append(matching)        

        return best_matching_target
        print('hello')
        # for coord in reference_corner:
        #     ssd = 0
        #     # for k, j in range(-search_size // 2, search_size // 2):
        #     reference_block = reference_img[
        #         max(0, coord[0] - block_size[0] // 2) : min(
        #             coord[0] + block_size[0] // 2, target_rows
        #         ),
        #         max(0, coord[1] - block_size[1] // 2) : min(
        #             coord[1] + block_size[1] // 2, target_cols
        #         ),
        #     ]
        #     for i, j in product(
        #         range(-search_block_size[0], search_block_size[0]),
        #         range(-search_block_size[1], search_block_size[1]),
        #     ):
        #         ssd = MAX_NUM
        #         x, y = coord[0] + i, coord[1] + j
        #         reference_block = reference_img[
        #             max(0, x - block_size[0] // 2) : min(
        #                 max(0, x + block_size[0] // 2), target_rows
        #             ),
        #             max(0, y - block_size[1] // 2) : min(
        #                 max(0, y + block_size[1] // 2), target_cols
        #             ),
        #         ]
        #         if reference_block.shape == target_block.shape:
        #             temp_ssd = np.sum((target_block-reference_block)**2)
        #             if temp_ssd < ssd:
        #                 ssd = temp_ssd
        #                 matching = (coord, [x, y])

        #     best_matching_target.append(matching)        
        # return best_matching_target

    def estimate_homography_2D(self, target_path, reference_path):
        target_img = cv2.imread(target_path)
        reference_img = cv2.imread(reference_path)
        putative_crsp = self.ssd_matching(target_img, reference_img)

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

        def Sampson_error(self, x1: np.ndarray, x1_e, x2: np.ndarray, x2_e):
            X_r = np.c_[x1, x2].transpose()
            X_e = np.c_[x1_e, x2_e].transpose()
