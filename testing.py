import networkx as nx
import numpy as np

G = nx.Graph()

feature_vector=np.array([1,7,22])

G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)

G.add_edge(1,2, { 'features': np.exp(np.random.normal(0,1) - np.random.normal(0,1)) })
G.add_edge(2,3, { 'features': np.exp(np.random.normal(0,1) - np.random.normal(0,1)) })
G.add_edge(2,4, { 'features': np.exp(np.random.normal(0,1) - np.random.normal(0,1)) })

personalize = { 1:10, 2:0, 3:0, 4:0 }
print(nx.pagerank(G, alpha=0.2, personalization=personalize, max_iter=100, tol=1.0e-6, nstart=None, weight='features'))
personalize = { 1:0, 2:10, 3:0, 4:0 }
print(nx.pagerank(G, alpha=0.2, personalization=personalize, max_iter=100, tol=1.0e-6, nstart=None, weight='features'))
personalize = { 1:0, 2:0, 3:10, 4:0 }
print(nx.pagerank(G, alpha=0.2, personalization=personalize, max_iter=100, tol=1.0e-6, nstart=None, weight='features'))
personalize = { 1:0, 2:0, 3:0, 4:100 }
print(nx.pagerank(G, alpha=0.2, personalization=personalize, max_iter=100, tol=1.0e-6, nstart=None, weight='features'))

G2 = nx.fast_gnp_random_graph(200,.8)
print(G2.edges())

for edge in G2.edges():
    G2[edge[0]][edge[1]]['weight'] = np.exp(np.random.normal(0,1) - np.random.normal(0,1))

for node in G2.nodes():
    personalize = dict((node, 0) for node in G2.nodes())
    personalize[node] = 10
