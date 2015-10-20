import networkx as nx
import numpy as np
from scipy.optimize import minimize

G2 = nx.fast_gnp_random_graph(200,.8)

for edge in G2.edges():
    feature1 = np.random.normal(0,1)
    feature2 = np.random.normal(0,1)
    G2[edge[0]][edge[1]]['features'] = [feature1, feature2]
    G2[edge[0]][edge[1]]['weight'] = np.exp(feature1 - feature2)

# positive training group
D = []
# negative training group
L = []
# Run personalized Page Rank to obtain p*
for node in G2.nodes():
    personalize = dict((node, 0) for node in G2.nodes())
    personalize[node] = 10
    p = nx.pagerank(G2, alpha=0.2, personalization=personalize, max_iter=100, tol=1.0e-6, nstart=None, weight='weight')
    for i in sorted(p.items(), key=lambda x: x[1], reverse=True):
        if not i[0] in G2.neighbors(node):
            D.append(i[0])
        else:
            L.append(i[0])
    D.remove(node)
    print(D)
    print(L)
    break

# This will become F(w) function
def f (w=2, reg=1, D=[], L=[]):
    return w**2 + reg * np.sum([1,2,3])

# This will become the partial derivative of F(w) wrt to w
def f_der(w):
    return 2*w + np.sum([1,2,3])

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der
# Minimize using BFGS to find optimal w
x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
res = minimize(rosen, x0, method='BFGS', jac=rosen_der, options={'disp': True})

print(res)
