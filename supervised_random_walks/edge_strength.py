import numpy as np

# Input feature vector psi (nxnxm), w (1xm) parameter vector
# Output Edge Strength matrix A (nxn)
# Basic edge strength = psi(i) * w(i)
# alpha(ij) = SUM (i = 1 to |w|) psi(i) * w(i)
def edge_strength(psi, w):
    alpha = np.multiply(psi, w).sum(axis=3)[0,:,:]
    return alpha

# Derivative of alpha wrt to w
# A: Edge strength matrix (nxn)
# psi: feature mastrix nxnxm
# w parameter vector: 1xm
# Output Edge strength derivative matrix (alpha_derivative) nxnxm
# Basic edge_strength_derivative is psi since partial derivative of  psi(i) * w(i) wrt w = psi(i)
def edge_strength_derivative(psi, w):
    alpha_derivative = psi
    return alpha_derivative
