import numpy as np
# Edge strength function

def edge_strength(psi, w):
    alpha = np.multiply(psi, w).sum(axis=3)[0,:,:]
    return alpha
