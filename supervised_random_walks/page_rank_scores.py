import numpy as np

# Initialize PageRank scores p

def initialize_page_rank(adjacency_matrix_cardinality):
    p = np.full([adjacency_matrix_cardinality], 1 / adjacency_matrix_cardinality)
    return p

def initialize_partial_derivatives(adjacency_matrix_cardinality, w_cardinality):
    dp = np.zeros((adjacency_matrix_cardinality, w_cardinality))
    return dp
