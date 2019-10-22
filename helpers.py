#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
helpers.py: Helper functions 
"""
import os

import matplotlib.pylab as plt


def make_dirs_safe(fname):
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def savefig(fname, fig=None):
    make_dirs_safe(fname)
    if fig is None:
        plt.savefig(fname, bbox_inches='tight')
    else:
        fig.savefig(fname, bbox_inches='tight')

    print('saved as', fname)
