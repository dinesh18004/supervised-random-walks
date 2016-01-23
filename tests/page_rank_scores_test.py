from nose.tools import *
import numpy as np
from supervised_random_walks.page_rank_scores import *
from supervised_random_walks.transition_matrix import *
from supervised_random_walks.edge_strength import *


def adjacency_matrix():
    am = [[0, 1, 1],
          [1, 0, 0],
          [1, 0, 0]]
    return am

def sample_edge_strength():
    return np.matrix([[0, 1, 8],
                [2, 0, 0],
                [2, 0, 0]])

def transition_matrix():
    stm = stochastic_transition_matrix(sample_edge_strength())
    return stm

def initialize_page_rank_test_returns_a_two_dimensional_array():
    v = initialize_page_rank(2) == np.array([0.5, 0.5])
    assert_equal(v.all(),True)

def initialize_partial_derivatives_test_returns_zeroes_matrix():
    v = initialize_partial_derivatives(2, 2) == np.array([[0, 0],[0, 0]])
    assert_equal(v.all(),True)

def test_page_rank_vector():
    #raise Exception(final_transition_matrix(transition_matrix(), alpha=0.3))
    Q = final_transition_matrix(transition_matrix(), alpha=0.3)
    p = page_rank_vector(adjacency_matrix(), Q)
    #raise Exception(p)
    #raise Exception(page_rank_vector(adjacency_matrix(), Q))

def test_page_rank_derivative():
    w = [1,1]
    psi =  np.array([[
                     [[0,0],[1,1],[1,1]],
                     [[1,1],[0,0],[1,1]],
                     [[1,1],[1,1],[0,0]]
                    ]])
    A = edge_strength(psi, w)
    dA = edge_strength_derivative(psi, w)
    Q_prime = stochastic_transition_matrix(A)
    Q = final_transition_matrix(Q_prime, alpha=0.3)
    Q_derivative = final_transition_matrix_derivative(Q, A, dA)
    p = page_rank_vector(adjacency_matrix(), Q)
    dP = page_rank_derivative(p, Q, Q_derivative, w)
    raise Exception(dP)



def test_coverged_returns_true_if_value_is_less_then_epsilon():
    assert_equal(converged(np.matrix([1]),np.matrix([1])), True)

def test_converged_returns_false_if_value_greater_then_epsilon():
    assert_equal(converged(np.matrix([1]), np.matrix([2])), False)
