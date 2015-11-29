# Takes edge strength matrix and returns transition matrix
import numpy as np

def stochastic_transition_matrix(A):
    row_sum = A.sum(axis=1)
    q_prime = np.divide(A, row_sum)
    return q_prime

def final_transition_matrix(Q_prime, alpha=0.2):
    Q = np.multiply(Q_prime,(1-alpha))
    # += 1*alpha restart probabilityx
    return Q
