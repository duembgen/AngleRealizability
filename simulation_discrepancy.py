#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulation_discrepancy.py: Create discrepancy results.
"""

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from pylocus.algorithms import procrustes
from pylocus.basics_angles import get_theta_tensor

from algorithms import reconstruct_from_angles, reconstruct_theta
from algorithms import solve_constrained_optimization
from angle_set import AngleSet
from angle_set import create_theta
from angle_set import get_index

COLUMNS = [
    'theta_sine', 'theta_sine_reconstructed', 'theta_noisy',
    'theta_noisy_reconstructed', 'error_sine', 'error_noisy', 'n_it', 'n_sine',
    'n_linear', 'n_total', 'success'
]


def mae(a, b):
    assert len(a.flatten()) == len(b.flatten())
    return np.sum(np.abs(a.flatten() - b.flatten())) / len(a.flatten())


def mse(a, b):
    assert len(a.flatten()) == len(b.flatten())
    return np.sum((a.flatten() - b.flatten())**2.0) / len(a.flatten())


def get_noisy(vec, scale=0.1):
    theta_noisy = vec.copy()
    theta_noisy += np.random.normal(scale=scale, loc=0, size=theta_noisy.shape)
    return theta_noisy


def inner_loop(angle_set, verbose=False, learned=False):
    df = pd.DataFrame(columns=COLUMNS)
    df_counter = 0

    n_rays = angle_set.get_n_rays()
    n_poly = angle_set.get_n_poly()
    n_linear_total = n_rays + n_poly
    n_sine_total = angle_set.get_n_sine()
    necessary = angle_set.M - angle_set.get_DOF()
    assert n_sine_total + n_linear_total == necessary, '{} {} {}'.format(
        necessary, n_linear_total, n_sine_total)
    print('number of sine constraints for N={}: {}'.format(
        angle_set.N, n_sine_total))

    # create noisy angle vector
    
    theta_noisy = get_noisy(angle_set.theta, scale, scale)

    # create linear constraints
    if not learned:
        Apoly, bpoly = angle_set.get_triangle_constraints()
        Arays, brays = angle_set.get_ray_constraints(verbose=False)
        Afull = np.vstack([Arays, Apoly[:n_poly]])
        bfull = np.hstack([brays, bpoly[:n_poly]])
    else:
        Afull, bfull = generate_linear_constraints(angle_set.points)
        if Afull.shape[0] != n_linear_total:
            raise RuntimeError('Did not learn enough linear constraints.')

    # reconstruct raw for baseline
    theta_noisy_reconstructed, points_noisy = reconstruct_theta(
        theta_noisy, angle_set.corners, angle_set.N)
    error_noisy = mse(points_noisy, angle_set.points)

    # first linear then sine.
    for n_linear in range(n_linear_total + 1):
        if n_linear < n_linear_total:
            n_sine_here = 0
        else:
            n_sine_here = n_sine_total

        for n_sine in range(n_sine_here + 1):
            if verbose and n_sine > 0:
                print('n_sine {}/{}'.format(n_sine, n_sine_total))
                print('n_total', n_total)

            choices_sine = range(n_sine)

            n_total = n_linear + n_sine

            choices_linear = range(n_linear)
            eps = 1e-10
            theta_sine, success = solve_constrained_optimization(
                theta_noisy,
                angle_set.corners,
                N=angle_set.N,
                Afull=Afull,
                bfull=bfull,
                choices_sine=choices_sine,
                choices_linear=choices_linear,
                eps=eps)
            if any(theta_sine == eps):
                raise RuntimeError('Found zero angle.')

            theta_sine_reconstructed, points_sine = reconstruct_theta(
                theta_sine, angle_set.corners, angle_set.N)

            points_sine, *_ = procrustes(
                angle_set.points, points_sine, scale=True)
            error_sine = mse(points_sine, angle_set.points)

            df.loc[df_counter, :] = {
                'theta_sine': theta_sine,
                'theta_sine_reconstructed': theta_sine_reconstructed,
                'theta_noisy': theta_noisy,
                'theta_noisy_reconstructed': theta_noisy_reconstructed,
                'n_sine': n_sine,
                'n_linear': n_linear,
                'n_total': n_total,
                'n_it': None,
                'N': None,
                'success': success,
                'error_sine': error_sine,
                'error_noisy': error_noisy
            }
            df_counter += 1
    return df


def generate_linear_constraints(points, verbose=False):
    """ Given point coordinates, generate angle constraints. """
    from scipy.linalg import null_space
    from angle_set import create_theta, get_n_linear, perturbe_points

    N, d = points.shape
    num_samples = get_n_linear(N) * 2

    if verbose:
        print('N={}, generating {}'.format(N, num_samples))

    M = int(N * (N - 1) * (N - 2) / 2)
    thetas = np.empty((num_samples, M + 1))
    for i in range(num_samples):
        points_pert = perturbe_points(points, magnitude=0.0001)
        theta, __ = create_theta(points_pert)
        thetas[i, :-1] = theta
        thetas[i, -1] = -1

    CT = null_space(thetas)
    A = CT[:-1, :].T
    b = CT[-1, :]
    return A, b


if __name__ == "__main__":
    from helpers import make_dirs_safe

    d = 2  # do not change.
    Ns = np.arange(4, 9)  #20)
    starti = 0
    endi = 20
    learned = False

    df = pd.DataFrame(columns=COLUMNS)
    if learned:
        fname = 'results/discrepancy_learned.pkl'
    else:
        fname = 'results/discrepancy.pkl'

    make_dirs_safe(fname)

    for N in Ns:
        print('N', N, Ns)
        angle_set = AngleSet(N=N, d=d)
        for i in range(starti, endi):
            print('i={}/{}'.format(i, endi))
            np.random.seed(i)

            # make sure this angle set has no (almost) zero angles.
            success = False
            for _ in range(10):
                angle_set.set_points(mode='random')
                if not any(np.abs(angle_set.theta) < 1e-3):
                    success = True
                    break
            if not success:
                print(
                    'WARNING: skipping i={} cause did not find good configuration'
                    .format(i))
                continue

            try:
                df_inner = inner_loop(angle_set, verbose=True, learned=learned)
            except RuntimeError:  # did not find enough constraints
                print('WARNING: skipping i={} cause of RuntimeError'.format(i))
                continue
            df_inner.loc[:, 'N'] = N
            df_inner.loc[:, 'n_it'] = i
            df = pd.concat([df, df_inner], ignore_index=True, sort=False)
            df.to_pickle(fname)
            print('saved intermediate to', fname)
