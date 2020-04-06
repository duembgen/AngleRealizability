#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
angle_set.py: Class to create point set and generate inner angles etc. 
"""

import itertools
from math import pi

import numpy as np
import matplotlib.pylab as plt
from scipy.special import binom

from pylocus.point_set import PointSet
from pylocus.basics_angles import get_inner_angle
from pylocus.basics_angles import get_index
from pylocus.basics_angles import get_theta_tensor
from pylocus.basics_angles import from_0_to_2pi

DEBUG = False


def get_n_rays(N):
    sum_ = 0
    for i in range(1, N - 2):  # goes to N-3
        sum_ += i
    return N * sum_


def get_n_poly(N):
    return int(binom(N - 1, 2))


def get_n_linear(N):
    return get_n_rays(N) + get_n_poly(N)


def perturbe_points(points, magnitude=0.01):
    direction = np.random.uniform(-1, 1, size=points.shape)
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    points += direction * magnitude
    return points


def create_theta(points):
    """
    Create inner angles vector (between 0 and pi)

    :param points: point coordinates (N x d)
    :return: thetas (length M), corners (size M x 3)
    """
    N = points.shape[0]
    M = int(N * (N - 1) * (N - 2) / 2)

    thetas = np.empty((M, ))
    corners = np.empty((M, 3))
    k = 0
    for triangle in itertools.combinations(np.arange(N), 3):
        for _ in range(3):
            triangle = np.roll(triangle, 1)
            corners[k, :] = triangle
            if np.allclose(points[triangle[1]], points[triangle[2]], atol=1e-10):
                thetas[k] = 0.0
            else:
                thetas[k] = get_inner_angle(points[triangle[0]], points[tuple(triangle[1:]), :])
            k += 1
        inner_angle_sum = thetas[k - 1] + thetas[k - 2] + thetas[k - 3]
        assert abs(inner_angle_sum - pi) < 1e-8, \
            'inner angle sum: {} {} {}'.format(
                triangle, inner_angle_sum - np.pi, (thetas[k - 1], thetas[k - 2], thetas[k - 3]))
    return thetas, corners


def combine_fullrank_matrix(A, A_add, b, b_add, print_out=False):
    """ Add rows as long as the resulting matrix is fullrank.
    """

    assert np.linalg.matrix_rank(A) == A.shape[0]

    counter = 0
    for i in range(A_add.shape[0]):
        A_addedrow = np.vstack((A, A_add[i, :]))
        if np.linalg.matrix_rank(A_addedrow) > np.linalg.matrix_rank(A):
            counter += 1
            A = A_addedrow
            b = np.r_[b, b_add[i]]
        # Visualization of dependent matrices.
        else:
            if print_out:
                alpha = np.linalg.lstsq(A.T, A_add[i, :].reshape(-1, 1))[0]
                alpha = alpha.reshape(1, -1)[0, :]
                alpha[alpha < 1e-12] = 0.0
                indices = np.where(alpha > 0.0)[0]
                plt.matshow(A)
                plt.title('lin indep. rows')
                plt.matshow(np.vstack((A_add[i, :], A_add[i, :])))
                plt.title('lin. dep. row: {}*{}'.format(indices, alpha[indices]))
    if (print_out):
        print('added {} linearly independent rows.'.format(counter))
    return A, b


def get_index(corners, i, jk):
    """ Find the index that angle with corners
    i, jk has inside the corners set.
    """
    if type(jk) == tuple:
        jk = list(jk)
    assert corners.shape[1] == 3
    sol1 = list(np.where(np.all(corners == [i] + jk, axis=1))[0])
    sol2 = list(np.where(np.all(corners == [i] + jk[::-1], axis=1))[0])
    sol = sol1 + sol2
    if len(sol) == 0:
        print('did not find anything for', corners, i, jk)
    assert len(sol) == 1
    return sol[0]


def get_ray_constraints(points, corners, theta, verbose=False):
    eps = 1e-13
    N = points.shape[0]

    As = []
    bs = []

    for corner in range(N):
        if verbose:
            print('corner', corner)
        x0 = points[corner]
        indices = list(range(N))
        indices.pop(corner)
        other_points = points[indices]

        absolute_angles = []
        for i, xi in zip(indices, other_points):
            vec = xi - x0
            abs_rad = np.arctan2(vec[1], vec[0])
            absolute_angles.append(from_0_to_2pi(abs_rad) * 180 / np.pi)
        order = np.argsort(absolute_angles).tolist()
        ordered_indices = [indices[o] for o in order] + [indices[order[0]]]
        if verbose:
            print(ordered_indices)

        for num in range(2, N):
            if verbose:
                print('number', num)
            for start in range(N - 1 - num):
                if verbose:
                    print('from', ordered_indices[start], 'to', ordered_indices[start + num])

                # first use outer angle
                indices = [get_index(corners, corner, (ordered_indices[start], ordered_indices[start + num]))]
                angles = [theta[indices[-1]]]

                # then use all inner angles.
                for j in range(start, start + num):
                    indices.append(get_index(corners, corner, (ordered_indices[j], ordered_indices[j + 1])))
                    angles.append(theta[indices[-1]])

                max_idx = np.argmax(angles)
                max_angle = np.max(angles)
                angles.pop(max_idx)
                actual_max_idx = indices.pop(max_idx)

                newline = np.zeros((len(theta)))

                sum_ = sum(angles)
                if sum_ > np.pi:
                    if abs(2 * np.pi - sum_ - max_angle) > eps:
                        print('FAILED 2pi', 2 * np.pi - sum_, max_angle)
                    else:
                        # using angle could lead to precision errors
                        # but is less complicated than indexing.
                        newline[indices] = 1.0
                        newline[actual_max_idx] = 1.0
                        b = 2 * np.pi
                else:
                    if abs(sum_ - max_angle) > eps:
                        print('FAILED sum', sum_, max_angle)
                    else:
                        newline[indices] = 1.0
                        newline[actual_max_idx] = -1.0
                        b = 0

                As.append(newline.tolist())
                bs.append(b)

    A = np.array(As)
    b = np.array(bs)
    return A, b


class AngleSet(PointSet):
    """ Class containing relative angles between points.

    :param N: Number of points.
    :param d: dimension of point set.

    :param self.num_angles: Number of angles.
    :param self.theta_tensor: Tensor of inner angles.
    :param self.theta: Vector of inner angles.
    :param self.corners: Matrix of corners corresponding to inner angles. Row (k,i,j) corresponds to theta_k(i,j).
    """
    def __init__(self, N, d):
        PointSet.__init__(self, N, d)

        num_triangles = self.N * (self.N - 1) * (self.N - 2) / 6

        self.num_angles = int(3 * num_triangles)
        self.theta = np.empty([
            self.num_angles,
        ])
        self.theta_tensor = np.empty([N, N, N])
        self.corners = np.empty([self.num_angles, 3])

    def init(self):
        PointSet.init(self)
        self.theta, self.corners = create_theta(self.points)
        self.theta_tensor = get_theta_tensor(self.theta, self.corners, self.N)

    def get_inner_angle(self, corner, other):
        return get_inner_angle(self.points[corner, :], (self.points[other[0], :], self.points[other[1], :]))

    def get_theta(self, i, j, k):
        return self.theta_tensor[i, j, k]

    def reconstruct_aloc(self, theta=None):
        from pylocus.algorithms import reconstruct_aloc
        if theta is not None:
            print('Warning: theta input to reconstruct_aloc is now ignored.')
        i = 0
        j = 1
        Pi = self.points[i, :]
        Pj = self.points[j, :]
        k = 2
        Pk = self.points[k, :]
        reconstruction = reconstruct_aloc(Pi, Pj, i, j, self.theta_tensor, Pk, k)
        return reconstruction

    def get_convex_polygons(self, m, print_out=False):
        """ Find out which polygons in point set are convex (using ground truth positions).

        :param m: size of polygons (number of corners)
        
        :return: (ordered) indices of all convex polygones of size m.
        """
        convex_polygons = []
        for corners in itertools.combinations(np.arange(self.N), m):
            p = np.zeros(m, np.uint)
            p[0] = corners[0]
            left = corners[1:]
            # loop through second corners
            for i, second in enumerate(corners[1:m - 1]):
                p[1] = second
                left = np.delete(corners, (0, i + 1))
                for j, last in enumerate(corners[i + 2:]):
                    left = np.delete(corners, (0, i + 1, j + i + 2))
                    p[-1] = last
                    # loop through all permutations of left corners.
                    for permut in itertools.permutations(left):
                        p[2:-1] = permut
                        sum_here = 0
                        # sum over all inner angles.
                        for k in range(m):
                            sum_here += self.get_inner_angle(p[1], (p[0], p[2]))
                            p = np.roll(p, 1)
                        sum_target = (m - 2) * pi
                        if (abs(sum_here - sum_target) < 1e-14 or abs(sum_here) < 1e-14):
                            if (print_out):
                                print("Convex polygon found:    ", p)
                            convex_polygons.append(p.copy())
                        elif (sum_here > sum_target):
                            if (print_out):
                                print("Warning: got into impossible case.")
        return convex_polygons

    def get_polygon_constraints(self, range_polygones=range(3, 5), print_out=False):
        """ Create all convex polygone constraints.

        :param range_polygones: list of numbers of polygones to test.
        
        :return A, b: the constraints on the theta-vector of the form A*theta = b
        """
        rows_A = []
        rows_b = []
        for m in range_polygones:
            if (print_out):
                print('checking {}-polygones'.format(m))
            polygons = self.get_convex_polygons(m)
            row_A, row_b = self.get_polygon_constraints_m(polygons, print_out)
            rows_A.append(row_A)
            rows_b.append(row_b)
        return np.vstack(rows_A), np.hstack(rows_b)

    def get_triangle_constraints(self, corner=0):
        """ Create linearly independent triangle constraints. """
        rows_A = []
        others = np.delete(range(self.N), corner)
        for pair in itertools.combinations(others, 2):
            row = np.zeros(self.num_angles)
            triangle = [corner, *pair]
            for _ in range(3):
                triangle = np.roll(triangle, 1)
                idx = get_index(self.corners, triangle[0], (triangle[1], triangle[2]))
                row[idx] = 1.0
            rows_A.append(row)
        A = np.vstack(rows_A)
        b = np.full(A.shape[0], np.pi)
        return A, b

    def get_ray_constraints(self, verbose=False):
        return get_ray_constraints(self.points, self.corners, self.theta, verbose=verbose)

    def get_polygon_constraints_m(self, polygons_m, print_out=False):
        """
        :param range_polygones: list of numbers of polygones to test.

        :return A, b: the constraints on the theta-vector of the form A*theta = b
        """
        rows_b = []
        rows_A = []
        m = len(polygons_m[0])
        rows_b.append((m - 2) * pi * np.ones(len(polygons_m), ))
        for p in polygons_m:
            row = np.zeros((self.theta.shape[0], ))
            for k in range(m):
                index = get_index(self.corners, p[1], (p[0], p[2]))
                row[index] = 1
                p = np.roll(p, 1)
            assert np.sum(row) == m
            rows_A.append(row)

        A = np.vstack(rows_A)
        b = np.hstack(rows_b)
        num_constraints = A.shape[0]
        corners = self.corners.reshape((1, -1))
        if (print_out):
            print('shape of A {}'.format(A.shape))
        return A, b

    def get_linear_constraints(self, full_rank=True):
        """ Generate linear constraints. """
        n_rays = self.get_n_rays()
        n_poly = self.get_n_poly()
        Apoly, bpoly = self.get_polygon_constraints([3])
        Aray, bray = self.get_ray_constraints()
        if full_rank:
            Afull = np.vstack([Aray, Apoly[:n_poly]])
            bfull = np.hstack([bray, bpoly[:n_poly]])
        else:
            Afull = np.vstack([Aray, Apoly])
            bfull = np.hstack([bray, bpoly])
        return Afull, bfull

    def perturbe_points(self, magnitude=0.1):
        self.points = perturbe_points(self.points, magnitude)
        self.init()

    def get_DOF(self):
        return int(self.N * self.d - self.d * (self.d + 1) / 2 - 1)

    def get_n_rays(self):
        return get_n_rays(self.N)

    def get_n_poly(self):
        return get_n_poly(self.N)

    def get_n_linear(self):
        return get_n_linear(self.N)

    def get_n_sine(self):
        if self.N < 4:
            return 0
        return sum(list(range(1, self.N - 2)))
