# Takes edge strength matrix and returns transition matrix
import numpy as np

# Input edge strength matrix A,
# Output stochastic transition matrix based on edge strengths
def stochastic_transition_matrix(A):
    row_sum = A.sum(axis=1)
    q_prime = np.divide(A, row_sum)
    return q_prime

# Input Q' and some alpha (Probablity of restart to node s)
# Output Q, transition matrix
def final_transition_matrix(Q_prime, alpha=0.2):
    Q = np.multiply(Q_prime, (1-alpha))
    return Q
