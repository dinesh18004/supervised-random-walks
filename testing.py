import networkx as nx
import numpy as np

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