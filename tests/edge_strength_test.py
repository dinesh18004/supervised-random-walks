from nose.tools import *
import numpy as np
from supervised_random_walks.edge_strength import *

# This will become our sample feature vector nxnxm
def psi():
    return np.array([[
                     [[1,1],[1,1]],
                     [[1,1],[1,1]]
                    ]])

def w_func():
    return np.array([2,1])

def edge_strength_test_returns_a_two_dimensional_array():
    v = edge_strength([[[[1,1]]]], [1,1])
    assert_equal(np.shape(v), (1,1))

def edge_strength_test_returns_a_sum_of_feature_strengths():
    v = edge_strength(psi(), w_func()) == np.array([[5,3],[3,7]])
    assert_equal(v.all(), True)
