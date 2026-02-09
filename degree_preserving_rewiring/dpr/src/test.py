from rewiring_functions import *
import networkx as nx
# from generate_graphs_itm import generate_graph 

# generate_graph('weibull', 5, 10000)
# G = nx.erdos_renyi_graph(25, 0.1, seed=42)
G = nx.erdos_renyi_graph(1000, 0.01)
# G = nx.erdos_renyi_graph(14, 0.2, seed=42)
print(G.number_of_edges())

G, r = rewire(G, -0.4, "graph", 10)

print(r)
