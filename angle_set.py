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
    if N < 3:
        return 0
    sum_ = 0
    for i in range(1, N - 2):  # goes to N-3
        sum_ += i
    return N * sum_


def get_n_poly(N):
    if N < 3:
        return 0
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
            if np.allclose(
                    points[triangle[1]], points[triangle[2]], atol=1e-10):
                thetas[k] = 0.0
            else:
                thetas[k] = get_inner_angle(points[triangle[0]],
                                            points[tuple(triangle[1:]), :])
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
                plt.title('lin. dep. row: {}*{}'.format(
                    indices, alpha[indices]))
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
                    print('from', ordered_indices[start], 'to',
                          ordered_indices[start + num])

                # first use outer angle
                indices = [
                    get_index(
                        corners, corner,
                        (ordered_indices[start], ordered_indices[start + num]))
                ]
                angles = [theta[indices[-1]]]

                # then use all inner angles.
                for j in range(start, start + num):
                    indices.append(
                        get_index(
                            corners, corner,
                            (ordered_indices[j], ordered_indices[j + 1])))
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


def get_numbers(N):
    """ Generate number of lin. independent single and polygon constraints. """
    sum_ = 0
    for i in range(1, N-2): # goes to N-3
        sum_ += i
    n_rays = N * sum_
    n_poly = int(binom(N-1, 2))
    return n_rays, n_poly


def get_linear_constraints(angle_set, full_rank=True):
    """ Generate linear constraints. """
    n_rays, n_poly = get_numbers(angle_set.N)
    Apoly, bpoly = angle_set.get_polygon_constraints([3])
    Aray, bray = angle_set.get_ray_constraints()
    if full_rank:
        Afull = np.vstack([Aray, Apoly[:n_poly]])
        bfull = np.hstack([bray, bpoly[:n_poly]])
    else:
        Afull = np.vstack([Aray, Apoly])
        bfull = np.hstack([bray, bpoly])
    return Afull, bfull


class AngleSet(PointSet):
    """ Class containing absolute/relative angles and linear constraints.

    :param self.theta: Vector of inner angles.
    :param self.corners: Matrix of corners corresponding to inner angles. Row (k,i,j) corresponds to theta_k(i,j).
    :param self.T: Number of triangles.
    :param self.M: Number of inner angles.
    :param self.C: Number of linear constraints.
    :param self.A: Matrix of constraints (self.C x self.M)
    :param self.b: Vector of constraints (self.C x 1)
    """

    def __init__(self, N, d):
        from scipy import special
        PointSet.__init__(self, N, d)
        self.T = self.N * (self.N - 1) * (self.N - 2) / 6
        self.M = int(3 * self.T)
        self.theta = np.empty([
            self.M,
        ])
        self.theta_tensor = np.empty([N, N, N])
        self.corners = np.empty([self.M, 3])
        self.abs_angles = np.empty([self.N, self.N])
        self.C = 0
        self.A = np.empty((self.C, self.M))
        self.b = np.empty((self.C, 1))

    def copy(self):
        new = PointSet.copy(self)
        new.theta = self.theta.copy()
        return new

    def init(self):
        PointSet.init(self)
        self.theta, self.corners = create_theta(self.points)
        self.theta_tensor = get_theta_tensor(self.theta, self.corners, self.N)

    def create_abs_angles_from_edm(self):
        rows, cols = np.indices((self.N, self.N))
        pi_pj_x = (self.points[rows, 0] - self.points[cols, 0])
        pi_pj_y = (self.points[rows, 1] - self.points[cols, 1])
        D = np.sqrt(
            np.sum((self.points[rows, :] - self.points[cols, :])**2, axis=2))
        cosine = np.ones([self.N, self.N])
        sine = np.zeros([self.N, self.N])
        cosine[D > 0] = pi_pj_x[D > 0] / D[D > 0]
        sine[D > 0] = pi_pj_y[D > 0] / D[D > 0]
        Dc = acos(cosine)
        for i in range(Dc.shape[0]):
            for j in range(Dc.shape[0]):
                if cosine[i, j] < 0 and sine[i, j] < 0:
                    # angle between pi and 3pi/2
                    Dc[i, j] = 2 * pi - Dc[i, j]
                if cosine[i, j] > 0 and sine[i, j] < 0:
                    # angle between 3pi/2 and 2pi
                    Dc[i, j] = 2 * pi - Dc[i, j]
        self.abs_angles = Dc

    def get_inner_angle(self, corner, other):
        return get_inner_angle(
            self.points[corner, :],
            (self.points[other[0], :], self.points[other[1], :]))

    def get_theta(self, i, j, k):
        return self.theta_tensor[i, j, k]

    def get_orientation(k, i, j):
        from pylocus.basics_angles import from_0_to_2pi
        """calculate angles theta_ik and theta_jk theta produce point Pk.
        Should give the same as get_absolute_angle! """
        theta_ij = own.abs_angles[i, j]
        theta_ji = own.abs_angles[j, i]

        # complicated
        xi = own.points[i, 0]
        xj = own.points[j, 0]
        yi = own.points[i, 1]
        yj = own.points[j, 1]
        w = np.array([yi - yj, xj - xi])
        test = np.dot(own.points[k, :] - own.points[i, :], w) > 0

        # more elegant
        theta_ik = truth.abs_angles[i, k]
        diff = from_0_to_2pi(theta_ik - theta_ij)
        test2 = (diff > 0 and diff < pi)
        assert (test == test2), "diff: %r, scalar prodcut: %r" % (
            diff, np.dot(own.points[k, :] - own.points[i, :], w))

        thetai_jk = truth.get_theta(i, j, k)
        thetaj_ik = truth.get_theta(j, i, k)
        if test:
            theta_ik = theta_ij + thetai_jk
            theta_jk = theta_ji - thetaj_ik
        else:
            theta_ik = theta_ij - thetai_jk
            theta_jk = theta_ji + thetaj_ik
        theta_ik = from_0_to_2pi(theta_ik)
        theta_jk = from_0_to_2pi(theta_jk)
        return theta_ik, theta_jk

    def return_noisy(self, noise, mode='normal', idx=0, visualize=False):
        if mode == 'normal':
            theta = self.theta.copy() + np.random.normal(0, noise, self.M)
            if (visualize):
                plot_thetas([self_theta, theta], ['original', 'noise'])
            return theta
        elif mode == 'constant':
            theta = self.theta.copy() + noise
            if (visualize):
                plot_thetas([self_theta, theta], ['original', 'noise'])
            return theta
        elif mode == 'punctual':
            theta = self.theta.copy()
            theta[idx] += noise
            if (visualize):
                plot_thetas_in_one([self.theta, theta], ['original', 'noise'])
            return theta
        else:
            NotImplementedError(mode)

    def reconstruct_aloc(self, theta):
        from pylocus.algorithms import reconstruct_aloc
        from pylocus.basics_angles import get_theta_tensor

        theta_tensor = get_theta_tensor(self.theta, self.corners, self.N)
        i = 0
        j = 1
        Pi = self.points[i, :]
        Pj = self.points[j, :]
        k = 2
        Pk = self.points[k, :]
        reconstruction = reconstruct_aloc(Pi, Pj, i, j, theta_tensor, Pk, k)
        return reconstruction

    def get_convex_polygons(self, m, print_out=False):
        """
        :param m: size of polygones (number of corners)
        
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
                        sum_theta = 0
                        # sum over all inner angles.
                        for k in range(m):
                            sum_theta += self.get_inner_angle(
                                p[1], (p[0], p[2]))
                            p = np.roll(p, 1)
                        angle = sum_theta
                        sum_angle = (m - 2) * pi
                        if (abs(angle - sum_angle) < 1e-14
                                or abs(angle) < 1e-14):
                            if (print_out):
                                print("convex polygon found:    ", p)
                            convex_polygons.append(p.copy())
                        #  elif (angle < sum_angle):
                        #  if (print_out): print("non convex polygon found:",p,angle)
                        elif (angle > sum_angle):
                            if (print_out):
                                print("oops")
        return convex_polygons

    def get_polygon_constraints(self,
                                range_polygones=range(3, 5),
                                print_out=False):
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
        self.A = np.vstack(rows_A)
        self.b = np.hstack(rows_b)
        return self.A, self.b

    def get_triangle_constraints(self, corner=0):
        """ Create linearly independent triangle constraints. """
        rows_A = []
        others = np.delete(range(self.N), corner)
        for pair in itertools.combinations(others, 2):
            row = np.zeros(self.M)
            triangle = [corner, *pair]
            for _ in range(3):
                triangle = np.roll(triangle, 1)
                idx = get_index(self.corners, triangle[0],
                                (triangle[1], triangle[2]))
                row[idx] = 1.0
            rows_A.append(row)
        A = np.vstack(rows_A)
        b = np.full(A.shape[0], np.pi)
        return A, b

    def get_indices(self, k):
        """ Get indices of theta vector that have k as first corner.
        
        :param k: Index of corner.

        :return indices_rays: Indices of ray angles in theta vector.
        :return indices_triangles: Indices of triangle angles in theta vector.
        :return corners_rays: List of corners of ray angles.
        :return angles_rays: List of corners of triangles angles.
        """
        indices_rays = []
        indices_triangles = []
        corners_rays = []
        angles_rays = []
        for t, triangle in enumerate(self.corners):
            if triangle[0] == k:
                indices_rays.append(t)
                corners_rays.append(triangle)
                angles_rays.append(self.theta[t])
            else:
                indices_triangles.append(t)
        np_corners_rays = np.vstack(corners_rays)
        np_angles_rays = np.vstack(angles_rays).reshape((-1, ))
        return indices_rays, indices_triangles, np_corners_rays, np_angles_rays

    def get_ray_constraints(self, verbose=False):
        return get_ray_constraints(
            self.points, self.corners, self.theta, verbose=verbose)

    def get_angle_constraints_m(self, polygons_m, print_out=False):
        rows = []
        m = len(polygons_m[0])
        # initialization to empty led to A being filled with first row of
        # currently stored A!
        A = np.zeros((1, self.M))
        b = np.empty((1, ))
        for p in polygons_m:
            if len(p) < 4:
                break
            if (print_out):
                print('sum of angles for p {}'.format(p))
            for j in p:
                if (print_out):
                    print('for corner {}'.format(p[0]))
                # for k in range(2, m-1): # how many angles to sum up.
                k = m - 2
                row = np.zeros(self.M)
                # outer angle
                for i in range(1, m - k):
                    sum_angles = 0
                    # inner angles
                    for l in range(i, i + k):
                        sum_angles += self.get_inner_angle(
                            p[0], (p[l], p[l + 1]))
                        index = get_index(self.corners, p[0], (p[l], p[l + 1]))
                        if (print_out):
                            print('+ {} (= index{}: {})'.format(
                                (p[0], (p[l], p[l + 1])), np.where(index),
                                self.corners[index, :]))
                        row[index] = 1
                    index = get_index(self.corners, p[0], (p[i], p[i + k]))
                    if (print_out):
                        print(' = {} (= index{}: {})'.format(
                            (p[0], (p[i], p[i + k])), np.where(index),
                            self.corners[index, :]))
                    row[index] = -1
                    rows.append(row)
                    if (print_out):
                        print('sum_angles - expected:{}'.format(
                            sum_angles -
                            self.get_inner_angle(p[0], (p[i], p[i + k]))))
                    if np.sum(np.nonzero(A)) == 0:
                        A = row
                    else:
                        A = np.vstack((A, row))
                p = np.roll(p, 1)
        if A.shape[0] > 0:
            b = np.zeros(A.shape[0])
        self.A = A
        self.b = b
        return A, b

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
        A_repeat = np.repeat(A.astype(bool), 3).reshape((1, -1))
        corners = self.corners.reshape((1, -1))
        corners_tiled = np.tile(corners, num_constraints)
        if (print_out):
            print('shape of A {}'.format(A.shape))
        if (print_out):
            print('chosen angles m={}:\n{}'.format(
                m, (corners_tiled)[A_repeat].reshape((-1, m * 3))))
        if (print_out):
            print('{}-polygones: {}'.format(m, rows_A))
        self.A = A
        self.b = b
        return A, b

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
