import numpy as np
# Sample edge strength function
# Must be differentiable wrt to w

def edge_strength(psi, w):
    alpha = np.multiply(psi, w)
    return alpha
