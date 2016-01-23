import numpy as np
from copy import copy

# Initialize PageRank scores p

def initialize_page_rank(adjacency_matrix_cardinality):
    p = np.full([adjacency_matrix_cardinality], 1 / adjacency_matrix_cardinality)
    return p

def initialize_partial_derivatives(adjacency_matrix_cardinality, w_cardinality):
    dp = np.zeros((adjacency_matrix_cardinality, w_cardinality))
    return dp

# Takes adjacency_matrix , Final Transition matrix Q
# Creates stationary page rank vector p
def page_rank_vector(adjacency_matrix, Q):
    t = 1
    n = len(adjacency_matrix)
    p = initialize_page_rank(n)
    c = False
    while (not c and t < 100):
        p_new = copy(p)
        t += 1
        for i in range(n):
            p_new[i] = np.sum(np.multiply(p, Q[:,i]))
        p_new = np.divide(p_new, np.sum(p_new))
        c = converged(p, p_new)
        p = p_new
    return p

def page_rank_derivative(p, Q, Q_derivative, w):
    n = len(Q)
    m = len(w)
    dP = np.zeros((n,m))
    c = False
    for j in range(m):
        k = 0
        dQj = Q_derivative[:,:,j]
        while (not c and k < 100):
            k = k + 1
            dpw_new = dP[:,j]
            for i in range(n):
                dpw_new[i] = np.sum((Q[:,i] * dP[:,j]) + (p * dQj[:,i]))
            temp = dP[:,j]
            dP[:,j] = dpw_new;
            c = converged(dP[:,j], temp)
    return dP

# Covergence method
# Takes two p stationary matrix and determines if they are the same based on epsilon
def converged(p1, p2, epsilon = 1e-12):
    return np.max(np.abs(p1 - p2)) <= epsilon
