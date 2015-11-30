import numpy as np

# Input feature vector psi, w parameter vector
# Output Edge Strength matrix A
def edge_strength(psi, w):
    alpha = np.multiply(psi, w).sum(axis=3)[0,:,:]
    return alpha
