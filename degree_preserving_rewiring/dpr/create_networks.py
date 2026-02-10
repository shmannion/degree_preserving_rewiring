import networkx as nx
import pickle

def create_network(dataset, headers):
    """
    Parameters:
    ----------
        dataset: str
            the file loc for the text file containing the graph edge list
        headers: int
            number of header lines in the file

    Returns:
    ---------
        G: networkx.Graph
            graph corresponding to the edgelist

    """

    with open(dataset) as d:
        j = 0
        while j < headers:
            next(d)
            j += 1
        G = nx.Graph()
        for line in d:
            l = line.strip().split()
            G.add_edge(l[0], l[1])

    G = nx.convert_node_labels_to_integers(G, first_label=0)
    return G

def Create_Network1(dataset): ##Use if file has one header
    data = dataset
    with open(data) as df:
        G = nx.Graph()
        next(df)
        for line in df:
            l = line.strip().split()
            G.add_edge(l[0], l[1])        

    G = nx.convert_node_labels_to_integers(G, first_label = 0)  
    return G

def Create_Network2(dataset): ##Use if file has 2 headers
    data = dataset
    with open(data) as df:
        G = nx.Graph()
        next(df)
        next(df)
        for line in df:
            l = line.strip().split()
            G.add_edge(l[0], l[1])        

    G = nx.convert_node_labels_to_integers(G, first_label = 0)  
    return G

