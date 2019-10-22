#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_angle_set.py: 
"""

from angle_set import *

if __name__ == "__main__":
    a = [[1, 2, 3], [2, 3, 1], [4, 4, 1]]
    assert get_index(np.array(a), 2, [3, 1]) == 1
    assert get_index(np.array(a), 2, [3, 1]) == 1
