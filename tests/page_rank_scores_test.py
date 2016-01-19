from nose.tools import *
import numpy as np
from supervised_random_walks.page_rank_scores import *
from supervised_random_walks.transition_matrix import *

def adjacency_matrix():
    am = [[0, 1, 1],
          [1, 0, 0],
          [1, 0, 0]]
    return am

def transition_matrix():
    stm = stochastic_transition_matrix(np.matrix([[0, 1, 2],
                                                  [1, 0, 2],
                                                  [2, 2, 0]]))
    return stm

def initialize_page_rank_test_returns_a_two_dimensional_array():
    v = initialize_page_rank(2) == np.array([0.5, 0.5])
    assert_equal(v.all(),True)

def initialize_partial_derivatives_test_returns_zeroes_matrix():
    v = initialize_partial_derivatives(2, 2) == np.array([[0, 0],[0, 0]])
    assert_equal(v.all(),True)

def test_page_rank_vector():
    Q = final_transition_matrix(transition_matrix(), alpha=0.3)
    p = page_rank_vector(adjacency_matrix(), Q)
    raise Exception(np.multiply(Q, p))
    #raise Exception(page_rank_vector(adjacency_matrix(), Q))

def test_coverged_returns_true_if_value_is_less_then_epsilon():
    assert_equal(converged(np.matrix([1]),np.matrix([1])), True)

def test_converged_returns_false_if_value_greater_then_epsilon():
    assert_equal(converged(np.matrix([1]), np.matrix([2])), False)
