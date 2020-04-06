#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulation_angles_distances.py: Create angle vs. distances recovery results.
"""

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from pylocus.algorithms import procrustes
from pylocus.algorithms import reconstruct_mds
from pylocus.basics_angles import get_theta_tensor

from angle_set import AngleSet
from algorithms import reconstruct_from_angles
from algorithms import reconstruct_theta
from algorithms import solve_constrained_optimization
from simulation_discrepancy import generate_linear_constraints
from simulation_discrepancy import mse


def add_edm_noise(edm, sigma=0.1):
    distances = np.sqrt(edm[np.triu(edm) > 0])
    noisy_distances = distances + np.random.normal(scale=sigma, size=distances.shape)

    noisy_edm = np.empty(edm.shape)
    np.fill_diagonal(noisy_edm, 0.0)
    noisy_edm[np.triu_indices(N, 1)] = noisy_distances**2
    noisy_edm += noisy_edm.T
    SNR = np.mean(distances**2) / sigma**2
    SNR_dB = 10 * np.log10(SNR)
    return noisy_edm, SNR_dB


def add_theta_noise(theta, sigma=0.1):
    noisy_theta = theta + np.random.normal(scale=sigma, size=theta.shape)
    SNR = np.mean(theta**2) / sigma**2
    SNR_dB = 10 * np.log10(SNR)
    return noisy_theta, SNR_dB


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run discrepancy tests.') 
    parser.add_argument('--sizes', metavar='Ns', type=int, nargs='+', default=[1, 5, 10, 15, 20],
                        help='square sizes')
    parser.add_argument('--num_it', metavar='num_it', type=int, default=20, 
                        help='number of iterations')
    args = parser.parse_args()

    num_it = args.num_it
    sizes = args.sizes

    N = 5
    d = 2
    eps = 1e-10
    num_sigma = 11
    #sigmas_angle = np.logspace(-3, 1, num_sigma)
    #sigmas_distance = [0.1, 1.0, 5.0]
    sigmas_angle = np.logspace(-5, 1, num_sigma)
    sigmas_distance = np.logspace(-3, 1, 5)
    sigmas_distance[-1] = 5
    sigmas_distance *= 2

    fname = 'results/angles_distances.pkl'

    angle_set = AngleSet(N, d)
    columns = ['sigma', 'SNR', 'error', 'type', 'N', 'n_it', 'size']
    df_results = pd.DataFrame(columns=columns)
    df_counter = 0

    for size in sizes:
        print('size', size, '/', sizes)
        for i in range(num_it):
            print('n_it', i, '/', num_it - 1)
            np.random.seed(i)
            angle_set.set_points('random', size=size)
            Afull, bfull = generate_linear_constraints(angle_set.points)
            n_sine = angle_set.get_n_sine()
            choices_linear = range(Afull.shape[0])
            choices_sine = range(n_sine)

            for j, sigma in enumerate(sigmas_distance):
                print('distance', j, '/', len(sigmas_distance) - 1)
                noisy_edm, SNR_edm = add_edm_noise(angle_set.edm, sigma=sigma)
                x_edm = reconstruct_mds(noisy_edm, all_points=angle_set.points)
                x_edm, *_ = procrustes(angle_set.points, x_edm)
                error_edm = mse(x_edm, angle_set.points)

                # effective noise is only sigma/2
                # because of symmetry of EDM.
                df_results.loc[df_counter, :] = dict(sigma=sigma / 2,
                                                     SNR=SNR_edm,
                                                     error=error_edm,
                                                     type='distance',
                                                     N=N,
                                                     n_it=i,
                                                     size=size)
                df_counter += 1

            for k, sigma in enumerate(sigmas_angle):
                print('angle', k, '/', len(sigmas_angle) - 1)
                noisy_theta, SNR_theta = add_theta_noise(angle_set.theta, sigma=sigma)
                denoised_theta, success = solve_constrained_optimization(noisy_theta,
                                                                         angle_set.corners,
                                                                         N=angle_set.N,
                                                                         Afull=Afull,
                                                                         bfull=bfull,
                                                                         choices_sine=choices_sine,
                                                                         choices_linear=choices_linear,
                                                                         eps=None)
                if any(denoised_theta == eps):
                    print('Found zero theta...')
                    continue
                try:
                    __, x_angle = reconstruct_theta(denoised_theta, angle_set.corners, angle_set.N)
                except RuntimeError:
                    print('RuntimeError...')
                    continue
                except AssertionError:
                    print('AssertionError...')
                    continue

                x_angle, *_ = procrustes(angle_set.points, x_angle)

                error_theta = mse(x_angle, angle_set.points)

                df_results.loc[df_counter, :] = dict(sigma=sigma,
                                                     SNR=SNR_theta,
                                                     error=error_theta,
                                                     type='angle',
                                                     N=N,
                                                     n_it=i,
                                                     size=size)
                df_counter += 1

            df_results.to_pickle(fname)
            print('saved intermediate results as', fname)
