import networkx as nx
import numpy as np
from scipy.optimize import minimize
# Training Function
# Input: Initial W, Training Nodes, Feature Vector
# w = initial w values
# C = [1xn] adjacency list for node s
# psi = Feature vector containing adjacency list features
# Returns w*, vector containing minimum w

class TrainingFunction(object):

    def __init__(self, w = [], C = [], psi = [[]]):
        self.w = w
        self.nodes = C
        self.psi = psi

    def initial_w(self):
        return self.w

    def training_nodes(self):
        return self.nodes

    def feature_vector(self):
        return self.psi
