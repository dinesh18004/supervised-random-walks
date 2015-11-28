# Takes edge strength matrix and returns transition matrix
import numpy as np

def transition_matrix(A):
    row_sum = A.sum(axis=1)
    q_prime = np.divide(A, row_sum)
    return q_prime
