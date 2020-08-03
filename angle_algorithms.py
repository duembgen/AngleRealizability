import itertools

import numpy as np

from pylocus.basics_angles import from_0_to_2pi
from pylocus.basics_angles import from_0_to_pi

from AngleRealizability import angle_set


def preprocess(alphas_per_drone):
    """ Convert to [0, pi] and sort the rays at each drone. """
    for drone, copy_alphas in alphas_per_drone.items():
        v_min = min(copy_alphas.values())
        alphas_per_drone[drone] = {k: from_0_to_2pi(v - v_min) for k, v in copy_alphas.items()}
        alphas_per_drone[drone] = {k: v for k, v in sorted(copy_alphas.items(), key=lambda item: item[1])}


def get_direction(alpha):
    alpha_rad = alpha / 180 * np.pi
    return np.array([np.cos(alpha_rad), np.sin(alpha_rad)])


def get_triangle_constraints(corners, verbose=False):
    A_rows = []
    available_nodes = [node for corner in corners for node in corner]
    available_nodes = sorted(np.unique(available_nodes))
    # instead of testing all possible combinations, it is enough
    # to look at all triangles involving the first corner.
    for other_two in itertools.combinations(available_nodes[1:], 2):
        triangle = [available_nodes[0]] + list(other_two)
        A_line = np.zeros(len(corners))
        assert triangle[1] <= triangle[2]  # corners are saved in ascending order

        indices = []
        invalid_triangle = False
        for _ in range(3):
            triangle = np.roll(triangle, 1)
            index = angle_set.get_index(corners, triangle[0], triangle[1:])
            if index is None:
                invalid_triangle = True
                break
            A_line[index] = 1
        if not invalid_triangle:
            A_rows.append(A_line)
    A = np.array(A_rows).reshape((-1, len(corners)))
    return A, np.ones((A.shape[0], )) * np.pi


def get_theta_from_alpha(alphas_per_drone):
    thetas = []
    corners = []
    for drone, alphas_dict in alphas_per_drone.items():
        others = sorted(alphas_dict.keys())
        for i, j in itertools.combinations(others, 2):
            assert j >= i
            thetas.append(from_0_to_pi(alphas_dict[i] - alphas_dict[j]))
            corners.append([drone, i, j])
    return np.array(thetas), np.array(corners)


def get_ray_constraints(alphas_dict, corner, corners, verbose=False):
    def get_idx(i, j, k):
        idx = angle_set.get_index(corners, i, (j, k))
        if idx is None:
            raise ValueError(i, j, k)
        return idx

    indices = list(alphas_dict.keys())
    values = list(alphas_dict.values())

    assert np.all(values == sorted(
        values)), f'Error in get_ray_constraints: values not sorted! Run preprocess before calling this function.'

    A_rows = []
    bs = []

    num_rays = len(alphas_dict)
    for num in range(2, num_rays):
        for i in range(num_rays - num):
            A_line = np.zeros(len(corners))

            # we can either have one bigger angle, and all
            # other angles sum up to this one bigger angle;
            # or all angles sum up to 2pi.
            # Below we determine the bigger angle, if it exists.
            b = 2 * np.pi
            for i1 in range(i, i + num):
                i2 = (i1 + 1)

                A_line[get_idx(corner, indices[i1], indices[i2])] = 1

                val = np.abs(values[i2] - values[i1])
                if val > np.pi:
                    A_line[get_idx(corner, indices[i1], indices[i2])] = -1
                    if b == 0:
                        raise ValueError('can only have one bigger angle')
                    b = 0

            # "close the loop"
            val = from_0_to_2pi(values[i] - values[i + num])
            A_line[get_idx(corner, indices[i], indices[i + num])] = 1
            if val > np.pi:
                A_line[get_idx(corner, indices[i], indices[i + num])] = -1
                if b == 0:
                    raise ValueError('can only have one bigger angle')
                b = 0

            A_rows.append(A_line)
            bs.append(b)
    A = np.array(A_rows)
    A = A.reshape((-1, len(corners)))
    b = np.array(bs).reshape((A.shape[0], ))
    return A, b


def test_feasibility(alphas_per_drone, verbose=False, nonlinear=False):
    """
    Use the realizability constraints from the ICASSP algorithm to test the feasibility
    of the angle set. 

    :param alphas_per_drone: dictionary of given angles. The structure is:
        {
          0:{ 
            1: abs_angle_01,
            2: abs_angle_02, 
            ...
          },
          1:{ 
            0: abs_angle_10,
            2: abs_angle_12,
            ... 
          }, 
          ...
        }
    :param nonlinear: whether or not to test the nonlinar (sine) constraitns. 
    """
    num_drones = len(alphas_per_drone)

    # create theta vector and ray constraints
    theta, corners = get_theta_from_alpha(alphas_per_drone)
    if len(theta) == 0:
        return 0, 0

    A = np.zeros((0, len(theta)))
    b = np.zeros((0, ))
    for drone, alphas_dict in alphas_per_drone.items():
        if len(alphas_dict) == 0:
            continue
        A_here, b_here = get_ray_constraints(alphas_dict, drone, corners, verbose=verbose)
        #ordered_indices = list(alphas_dict.keys())
        #A_here1, b_here1 = angle_set.define_ray_constraints(drone, corners, theta, ordered_indices)
        #A_here1 = np.array(A_here1)
        #print('b error', b_here - b_here1)
        #print('A error', np.where(A_here==1)[0], np.where(A_here1==1)[0])
        #print('A error', np.where(A_here==-1)[0], np.where(A_here1==-1)[0])

        A = np.r_[A, A_here]
        b = np.r_[b, b_here]
    if verbose:
        print('number of rays constraints', A.shape[0], angle_set.get_n_rays(num_drones))
        print('rays error:', np.sum(np.abs(A.dot(theta) - b)))
    A_tri, b_tri = get_triangle_constraints(corners, verbose=verbose)
    A = np.r_[A, A_tri]
    b = np.r_[b, b_tri]
    if verbose:
        print('number of triangle constraints', A_tri.shape[0], angle_set.get_n_poly(num_drones))
        print('triangle error:', np.sum(np.abs(A_tri.dot(theta) - b_tri)))

    mae = np.sum(np.abs(A.dot(theta) - b))
    if len(b):
        mae /= len(b)
    return mae, A.shape[0]
