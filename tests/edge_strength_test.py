from nose.tools import *
import numpy as np
from supervised_random_walks.edge_strength import *

# This will become our sample feature vector nxnxm
def psi():
    return np.array([[
                     [[2,1],[1,1]],
                     [[1,1],[2,3]]
                    ]])

def w_func():
    return np.array([2,1])

def edge_strength_test():
    v = edge_strength(psi(), w_func()) == np.array([[[4,1],[2,1]],[[2,1],[4,3]]])
    assert_equal(v.all(), True)
