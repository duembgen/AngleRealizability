import copy
import enum
import itertools

import numpy as np

from pylocus.basics_angles import from_0_to_pi

from AngleRealizability import angle_set
from angle_algorithms import test_feasibility
from angle_algorithms import preprocess

VERBOSE = 0  # only found solution and errors
#VERBOSE = 1  # debugging
#VERBOSE = 2 # verbose debugging


class SolutionStatus(enum.Enum):
    rejected = 0  # Partial solution is rejected because not feasible.
    accepted = 1  # Partial solution passes all tests. Print out as possible solution.
    unknown = 2  # Partial solution passes tests so far. Continue with it.


class PartialSolution(object):
    def __init__(self):
        self.edges = []
        self.indices = []
        self.values = []
        self.nodes = [0]
        self.idx = -1  # pointer to current index.

    def __len__(self):
        return self.idx + 1

    def __str__(self):
        return str(np.round(self.values[:self.idx + 1], 2))

    def copy(self):
        return copy.deepcopy(self)

    def update_nodes(self):
        nodes = [node for edge in self.edges[:self.idx + 1] for node in edge]
        self.nodes = np.unique(nodes).tolist()

    def add_new_edge(self, P, new_edge):
        self.edges.append(new_edge)
        self.values.append(P[new_edge[0]][new_edge[1]][0])
        self.indices.append(0)
        self.idx += 1

    def create_solution(self, P, indices, num_drones=3):
        self.edges = list(itertools.permutations(range(num_drones), 2))
        assert len(indices) == len(self.edges), f'need to give {len(self.edges)} indices!'
        self.indices = indices
        self.values = [P[e[0]][e[1]][i] for i, e in zip(indices, self.edges)]
        self.idx = len(indices) - 1
        self.nodes = list(range(num_drones))

    def assert_values(self, P):
        for v, e, idx in zip(self.values, self.edges, self.indices):
            assert v == P[e[0]][e[1]][idx]


def constraints_check(P, c, eps=1e-3):
    """
    check if constraints for partial solution c are met.
    
    :return SolutionStatus: 
        see SolutionStatus documentation for meanings.

    returns True if it is not clear yet, or ok.
    returns False if the angles in set do not satisfy constraint.
    """
    if VERBOSE > 0:
        print(f'checking next combination: {c.indices}(len {len(c.indices)}) up to {c.idx}')

    c.update_nodes()
    if len(c.nodes) < 3:
        return SolutionStatus.unknown, 'not enough corners'

    # TODO continue here: should only check everything up to index?
    alphas = {}
    for e, val in zip(c.edges, c.values):
        if alphas.get(e[0], False):
            alphas[e[0]][e[1]] = val
        else:
            alphas[e[0]] = {}
            alphas[e[0]][e[1]] = val

    preprocess(alphas)
    mae, counter = test_feasibility(alphas, verbose=VERBOSE > 0)
    num_drones = len(P)
    if VERBOSE > 1:
        print('MAE', mae, counter)

    number = angle_set.get_n_linear(num_drones)

    if counter == 0:
        return SolutionStatus.unknown, 'no mae performed yet'
    elif counter >= number and mae < eps:
        return SolutionStatus.accepted, f'{counter}>={number} mae passed.'
    elif mae < eps:
        return SolutionStatus.unknown, f'{counter}<{number} mae passed.'
    else:
        return SolutionStatus.rejected, f'{counter} mae failed.'


def root(P):
    return PartialSolution()


def next(P, c):
    """ Alternate choice for latest edge.
    """
    curr_edge_idx = c.idx
    curr_edge = c.edges[curr_edge_idx]
    if VERBOSE > 1:
        print('NEXT: current edge', curr_edge)

    available_values = P[curr_edge[0]][curr_edge[1]]
    curr_idx = c.indices[c.idx]
    curr_idx += 1

    if curr_idx == len(available_values):
        # go up in tree by one.
        if VERBOSE > 1:
            print('NEXT: going up in tree')
        c.idx = curr_edge_idx - 1
        return None  # return None: means we call first now.
    elif curr_idx < len(available_values):  # choose next for current index.
        c.values[c.idx] = available_values[curr_idx]
        c.indices[c.idx] = curr_idx
        return c
    else:
        ValueError(curr_idx, available_values)


def first(P, c):
    """ Add new edges to current solution c.
    
    return None if c has no more children and we have "closed
    the loop" already.
    """
    num_drones = len(P.keys())

    curr_edge_idx = c.idx  #c.edges[-1]

    if (curr_edge_idx == 0) and (VERBOSE > 1):
        print('FIRST: we are at root again')
    if (curr_edge_idx == -1) and (VERBOSE > 1):
        print('FIRST: we are at root for the first time')
    if curr_edge_idx < -1:  # we arrived at root, no more options.
        return None

    ###### we already have the next edge in storage.
    if curr_edge_idx < (len(c.edges) - 1):
        c.idx = curr_edge_idx + 1

        # start indexing from zero!
        c.indices[c.idx] = 0

        # set value correspondingly
        # TODO move this to partial_solution?
        curr_edge = c.edges[c.idx]
        available_values = P[curr_edge[0]][curr_edge[1]]
        c.values[c.idx] = available_values[0]
        return c

    ###### we do not have the next edge in storage.
    if len(c.edges) > 0:
        curr_edge = c.edges[curr_edge_idx]
        e1, e2 = curr_edge
    else:  # this is the very first edge
        e1 = 1
        e2 = 0

    # first check if we can add the "reverse" of this edge
    if [e2, e1] not in c.edges:
        new_edge = [e2, e1]

    # otherewise create a new edge according to our rule.
    else:
        e2, e1 = sorted([e1, e2])
        if (e2 == 0) and (e1 + 1 < num_drones):
            new_edge = [e1 + 1, e1]
            # this is a new node, so add it to the nodes list.
            # c.nodes.append(e1 + 1)
        elif (e2 == 0) and (e1 + 1 >= num_drones):  # arrived at leave
            return None
        elif e2 > 0:
            new_edge = [e1, e2 - 1]
    if VERBOSE > 1:
        print('FIRST: adding new edge', new_edge)
    c.add_new_edge(P, new_edge)
    return c


def backtrack(P, c):
    c.assert_values(P)
    status, message = constraints_check(P, c)
    if status == SolutionStatus.accepted:
        print('valid solution:', c.indices[:c.idx + 1], message)
    elif status == SolutionStatus.rejected:
        return None

    s = first(P, c)
    while s is not None:
        backtrack(P, s)
        s = next(P, s)
    return None


if __name__ == "__main__":
    pass
