from nose.tools import *
import numpy as np
from supervised_random_walks.page_rank_scores import *

def initialize_page_rank_test_returns_a_two_dimensional_array():
    v = initialize_page_rank(2) == np.array([0.5, 0.5])
    assert_equal(v.all(),True)

def initialize_partial_derivatives_test_returns_zeroes_matrix():
    v = initialize_partial_derivatives(2, 2) == np.array([[0, 0],[0, 0]])
    assert_equal(v.all(),True)
