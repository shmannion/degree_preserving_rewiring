from rewiring_functions import *
import networkx as nx
import pandas as pd
from generate_graphs_itm import generate_graph 

# for dist in ['weibull', 'lognormal', 'exponential']:
#     for i in range(0,30):
#         G = generate_graph(dist, 5, 10000)
#         G, r = rewire(G, -0.4, "graph", 10, method='max')
#         G, r = rewire(G, 0.4, "graph", 10, method='max')
# for i in range(0,50):
#     G = nx.erdos_renyi_graph(10000, 0.002)
#     G, r = rewire(G, -0.4, "graph", 10, method='max')
#     G, r = rewire(G, 0.4, "graph", 10, method='max')

# for i in range(0,50):
#     G = nx.barabasi_albert_graph(10000, 3)
#     G, r = rewire(G, -0.4, "graph", 10, method='max')
#     G, r = rewire(G, 0.4, "graph", 10, method='max')
# G = nx.erdos_renyi_graph(25, 0.1, seed=42)
# G = nx.erdos_renyi_graph(1000, 0.01, seed=42)
# G = nx.erdos_renyi_graph(14, 0.2, seed=42)
edges = pd.read_csv('../../../data/deezer_clean_data/HR_edges.csv', index_col=False)
G = nx.from_pandas_edgelist(edges, source='node_1', target='node_2', create_using=nx.Graph())
print(G.number_of_edges())

G, r = rewire(G, -0.4, "graph", 10, method='original')

print(r)
