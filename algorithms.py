#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
algorithms.py: Functions to solve angle-based point recovery problems. 
"""

import itertools
from math import pi, sin, floor

import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize

from pylocus.basics_angles import get_inner_angle
from pylocus.basics_angles import get_theta_tensor
from angle_set import create_theta

from angle_set import get_index

def get_angles(x, corn, i):
    """ Return the angles a, b, c, d, e, f
    in a quadrilateral satisfying:

    sin(c) * sin(b) * sin(f) =  sin(a) * sin(d) * sin(e)

    :param x: theta vector (M,)
    :param c: corners matrix (M, 3)
    :param i: 4 indices to use for sine constraints.
    """
    a = x[get_index(corn, i[2], [i[0], i[1]])]
    b = x[get_index(corn, i[2], [i[0], i[3]])]
    c = x[get_index(corn, i[1], [i[0], i[2]])]
    d = x[get_index(corn, i[3], [i[0], i[2]])]
    e = x[get_index(corn, i[1], [i[0], i[3]])]
    f = x[get_index(corn, i[3], [i[0], i[1]])]
    return a, b, c, d, e, f


def constraint_sine(x, corn, i):
    """
    :param x: theta vector (M,)
    :param c: corners matrix (M, 3)
    :param i: 4 indices to use for sine constraints.
    """
    a, b, c, d, e, f = get_angles(x, corn, i)
    lhs = sin(c) * sin(b) * sin(f)
    rhs = sin(a) * sin(d) * sin(e)
    return lhs - rhs


def constraint_sine_multi(x, c, N, choices=[]):
    """
    :param x: theta vector (M,)
    :param c: corners matrix (M, 3)
    :param N: number of points
    :param choices: list of constraints to add. The constraints are indexed from 0 to K-1, where K is the total number of available constraints. Per quadrilateral, we add one constraint. 
    """
    all_combinations = list(itertools.combinations(range(N), 4))
    # Also add one permutation per quadrilateral. This was once thought to be
    # necessary but it isn't.
    # all_combinations = all_combinations + [np.roll(arr, 1) for arr in all_combinations]

    sum_ = 0
    for choice in choices:
        if choice >= len(all_combinations):
            raise ValueError('higher choice than possible: {}'.format(
                len(all_combinations)))
        i = np.array(all_combinations[choice])
        sum_ += constraint_sine(x, c, i)
    return sum_ / len(choices)


def solve_constrained_optimization(theta_noisy,
                                   corners,
                                   Afull,
                                   bfull,
                                   N,
                                   choices_sine=[],
                                   choices_linear=[],
                                   eps=1e-10):
    """ Solve angle denoising with linear and nonlinear constraints.

    :param theta_noisy: noisy angle vector
    :param corners: corresponding corners matrix
    :param Afull: matrix of linear constraints
    :param bfull: vector of linear constraints.
    :param N: number of points.
    :param choices_sine: list of indices of sine constraints to impose. 
    :param choices_linear: list of indices of linear constraitns to impose (between 0 and Afull.shape[0]) 
    :param eps: if given, impose constraints on each element [eps, pi-eps]. If None, don't impose constraints.

    :return: denoised angle vector, success boolean.
    """

    def loss(x):
        return 0.5 * np.linalg.norm(theta_noisy - x)**2

    if eps is not None:
        bounds = np.c_[np.ones(theta_noisy.shape) * eps,
                       np.ones(theta_noisy.shape) * pi - eps]
    else:
        bounds = np.c_[-np.ones(theta_noisy.shape) * pi,
                       np.ones(theta_noisy.shape) * pi]
    cons = []

    # choose linear constraints
    if len(choices_linear) > 0:
        Apart = Afull[choices_linear]
        bpart = bfull[choices_linear]
        cons.append({
            'type': 'eq',
            'fun': lambda x: np.dot(Apart, x) - bpart,
            'jac': lambda x: Apart
        })
    # choose sine constraints
    if len(choices_sine) > 0:
        cons.append({
            'type':
            'eq',
            'fun':
            lambda x: constraint_sine_multi(x, corners, N, choices_sine)
        })

    # solve.
    options = {
        'disp': False,
        'ftol': 1e-10,
        'maxiter': 400
    }  # ftol: stopping crit. for SLSQP method
    res = minimize(
        loss,
        x0=theta_noisy,
        bounds=bounds,
        method='SLSQP',
        constraints=cons,
        options=options)
    theta_hat = res.x

    # make sure theta is bettwen 0 and pi.
    if eps is None:
        theta_hat = np.mod(np.abs(theta_hat), 2 * pi)
        theta_hat = np.minimum(2 * pi - theta_hat, theta_hat)
        assert np.all(theta_hat >= 0)
        assert np.all(theta_hat <= np.pi)
    return theta_hat, res.success


def normal(alpha):
    return np.array([np.cos(alpha), np.sin(alpha)]).reshape((2, 1))


def find_third_point(p0, p1, theta0, theta1, side=1):
    v = p1 - p0
    alpha_01 = np.arctan2(v[1], v[0])
    alpha_10 = np.arctan2(-v[1], -v[0])
    if side == 1:
        alpha_02 = alpha_01 + theta0
        alpha_12 = alpha_10 - theta1

        n0 = normal(alpha_02)
        n1 = normal(alpha_12)
    elif side == -1:
        alpha_02 = alpha_01 - theta0
        alpha_12 = alpha_10 + theta1

        n0 = normal(alpha_02)
        n1 = normal(alpha_12)

    A = np.r_[np.c_[n0, np.zeros((2, 1)), -np.eye(2)], np.c_[np.zeros(
        (2, 1)), n1, -np.eye(2)]]
    b = -np.r_[p0, p1]
    try:
        x = np.linalg.solve(A, b)
    except:  # SingularMatrix error.
        return None
    if np.any(np.isnan(x)):
        return None
    return x[2:]


def find_ith_point(p0, p1, p2, theta0_13, theta1_03, theta2_03):
    """ Find point of intersection consistent with theta2. """

    p3_this = find_third_point(p0, p1, theta0_13, theta1_03, side=1)
    p3_other = find_third_point(p0, p1, theta0_13, theta1_03, side=-1)
    if (p3_this is None) or (p3_other is None):
        return None

    theta2_this = get_inner_angle(p2, (p0, p3_this))
    theta2_other = get_inner_angle(p2, (p0, p3_other))

    # If these two thetas are very similar, then point p2
    # was not a good choice to solve the ambiguity.
    if abs(theta2_this - theta2_other) < 1e-10:
        return None

    if abs(theta2_this - theta2_03) < abs(theta2_other - theta2_03):
        return p3_this
    else:
        return p3_other


def reconstruct_theta(theta, corners, N):
    """ Given a theta vector, do simple build-up algorithm to 
    generate point set and reconstructed theta. 
    
    :param theta: vector of M angles
    :param corners: corresponding corners
    :param N: number of points

    :return: vector of M reconstructed angles, point set in canonical shape.
    """
    theta_tensor = get_theta_tensor(theta, corners, N)
    points_sine = reconstruct_from_angles(theta_tensor)
    theta_recon, c = create_theta(points_sine)
    return theta_recon, points_sine


def reconstruct_from_angles(theta_tensor, d=2):
    """ Build-up algorithm from inner angles. """
    import itertools
    N = theta_tensor.shape[0]
    points = np.empty((N, d))

    # fix first point at origin (fixes translation)
    points[0, :] = np.zeros(d)

    # fix second point on xaxis (fixes scale and orientation)
    points[1, :] = np.zeros(d)
    points[1, 0] = 1

    # find next point given 2 angles (fixes flip)
    theta0_12 = theta_tensor[0, 1, 2]
    theta1_02 = theta_tensor[1, 0, 2]
    point = find_third_point(points[0, :], points[1, :], theta0_12, theta1_02)
    if point is not None:
        points[2, :] = point
    else:
        print(theta0_12, theta1_02)
        raise RuntimeError('Degenerate starting points')

    # find next points given 3 angles (fully determined)
    for i in range(3, N):
        p_i = None
        counter = 0
        # we try all sorts of combinations of previous points, until we
        # find a non-ambiguous one. Usually only one iteration is necessary.
        candidates = list(itertools.permutations(range(i), 3))[::-1]
        while p_i is None:
            counter += 1
            if counter >= len(candidates):
                print('current points:', points)
                print('current thetas:', theta0_13, theta1_03, theta2_03)
                print('current theta tensor:', theta_tensor)
                raise RuntimeError('Degenerate configuration.')
            indices = candidates[counter]
            theta0_13 = theta_tensor[indices[0], indices[1], i]
            theta1_03 = theta_tensor[indices[1], indices[0], i]
            theta2_03 = theta_tensor[indices[2], indices[0], i]
            thetas = np.array([theta0_13, theta1_03, theta2_03])
            if np.any(np.abs(thetas) <= 1e-5):
                continue
            p_i = find_ith_point(
                points[indices[0], :],  #0
                points[indices[1], :],  #1
                points[indices[2], :],  #2
                theta0_13,
                theta1_03,
                theta2_03)
        if counter > 1:
            print('Warning: possibly degenerate configuration.')
        points[i, :] = p_i
    return points


if __name__ == "__main__":
    from angle_set import AngleSet
    from pylocus.algorithms import procrustes

    d = 2
    N = 5
    np.random.seed(51)
    angle_set = AngleSet(N=N, d=d)
    angle_set.set_points(mode='random')

    points = reconstruct_from_angles(angle_set.theta_tensor)
    points_fitted, *_ = procrustes(angle_set.points, points, scale=True)

    plt.figure()
    plt.scatter(*points_fitted.T, label='fitted')
    plt.scatter(*angle_set.points.T, label='original', marker='x')
    plt.axis('equal')
    plt.legend(loc='best')
    plt.show()
    #assert np.allclose(points_fitted, angle_set.points)
