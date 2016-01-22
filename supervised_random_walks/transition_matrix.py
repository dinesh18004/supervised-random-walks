# Takes edge strength matrix and returns transition matrix
import numpy as np
import numpy.matlib
# Input edge strength matrix A,
# Output stochastic transition matrix based on edge strengths
def stochastic_transition_matrix(A):
    row_sum = A.sum(axis=1)
    q_prime = np.divide(A, row_sum)
    return q_prime

# Input Q' and some alpha (Probablity of restart to node s)
# Output Q, transition matrix
def final_transition_matrix(Q_prime, alpha=0.2):
    Q = np.multiply(Q_prime, (1 - alpha))
    return Q

# Input A, A, and dA
# Output derivative of Q wrt to w
def final_transition_matrix_derivative(Q, A, dA, alpha = 0.2):
    n = len(A) # number of nodes
    m = dA.shape[3] # number of features
    sum_edge_strength = A.sum(axis=1)
    denominator = np.square(sum_edge_strength)
    denominator = np.matlib.repmat(denominator, 1, n)
    rep_sum_edge_strength = np.matlib.repmat(sum_edge_strength, 1, n)
    Q_derivative = np.zeros((n,n,m))
    for i in range(m):
        sum_edge_strength_derivative = np.transpose(dA[:,:,i].sum(axis=2))
        #raise Exception(sum_edge_strength_derivative)
        rep_sum_edge_strength_derivative = np.matlib.repmat(sum_edge_strength_derivative, 1, n)
        #raise Exception(rep_sum_edge_strength_derivative)
        temp = (rep_sum_edge_strength * dA[0][:,:,i]) - (A * rep_sum_edge_strength_derivative)
        #raise Exception(temp)
        Q_derivative[:,:,i] = temp / denominator

    return Q_derivative
