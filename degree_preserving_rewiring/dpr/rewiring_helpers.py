# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:39:46 2023

@author: shane
"""

import networkx as nx
import numpy as np
import pandas as pd
import time
import random


def degree_list(G):
    """
    Parameters
    ----------

    G : networkx.Graph OR list


    Returns
    -------
    np.ndarray
        sorted array of degrees

    """
    if type(G) == nx.classes.graph.Graph:
        degree_dict = dict(G.degree())
        degree_list = list(degree_dict.values())
    else:
        degree_list = G
    degree_list.sort()

    return np.array(degree_list)

def check_new_edges(potential_edges, G, row):
    """
    Takes the edges that will be potentially added to the Graph and checks
    for any issues
    Parameters
    ---------
    potential_edges : list of lists
        the edges to be checked
    
    G : networkx.Graph
        graph needed to check if any potential edges exist already

    row : the row to be added to the results DataFrame is edited here

    Returns
    -------
    edges_to_add : list of lists
        the checked edges

    row : dict
        the information to go into the results DataFrame

    """
    edges_to_add = []
    for edge in potential_edges:
        if G.has_edge(edge[0], edge[1]) == False:
            if edge[0] != edge[1]:
                if [edge[1], edge[0]] not in potential_edges:
                    if potential_edges.count(edge) == 1:
                        edges_to_add.append(edge)
                    else:
                        row['duplicate_edges'] += 0.5
                else:
                    row['duplicate_edges'] += 0.5
            else:
                row['self_edges'] += 1
        else:
            row['existing_edges'] += 1

    return edges_to_add, row 


def test_sample_sizes(G, name, sample_size, results=None, n_tests=1000):
    """
    Function to test the success rate of a given sample size on a graph.
    Takes a Graph, its name, the sample size to test. Optionally a results df and the number of tests (default 1000)

    returns dataframe that is just one row, n_successes, n_failures, and each kind of failure
    """

    j = 0
    row = {'name': name,
           'N': len(G.nodes()),
           'k': nx.average_degree_connectivity(G),
           'density': nx.density(G),
           'sample_size': sample_size, 
           'duplicate_edges': 0, 
           'self_edges': 0,
           'existing_edges': 0,
           'failures': 0,
           'successes': 0,
           'p_success': 0} 
    
    while j < n_tests:
        #define dictionary to track relevant info for each loop

        edges = list(G.edges())                
        edges_to_remove = random.sample(edges, sample_size)
        deg_dict = {}
        nodes = []
        for edge in edges_to_remove:
            for node in edge:
                nodes.append(node)
                deg_dict[node] = G.degree(node)
    
        nodes_sorted = sorted(nodes, key=deg_dict.get)
        potential_edges = [[nodes_sorted[i], nodes_sorted[i+1]] for i in range(0,len(nodes_sorted),2)]
        edges_to_add, row = check_new_edges(potential_edges, G, row)
        if len(edges_to_add) == len(potential_edges):
            row['successes'] += 1
        else:
            row['failures'] += 1

        # results.loc[len(results)] = row
        j += 1

    row['p_success'] = row['successes']/n_tests
    if results != None:
        results.loc[len(results)] = row
    else:
        results = pd.DataFrame([row])

    return results
