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


def test_sample_sizes(G, name, sample_size, direction,results=None, n_tests=1000):
    """
    Function to test the success rate of a given sample size on a graph.
    Takes a Graph, its name, the sample size to test. 
    Requires the direction to attempt: 'positive' or 'negative'
    Optionally a results df and the number of tests (default 1000)

    returns dataframe that is just one row, n_successes, n_failures, and each kind of failure
    """

    j = 0
    row = {'name': name,
           'N': len(G.nodes()),
           'k': nx.average_degree_connectivity(G),
           'density': nx.density(G),
           'sample_size': sample_size, 
           'direction': direction,
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
        if direction == 'positive':
            potential_edges = [[nodes_sorted[i], nodes_sorted[i+1]] for i in range(0,len(nodes_sorted),2)]
        else:
            n_nodes = int(len(nodes_sorted)/2)
            potential_edges = [(nodes_sorted[i], nodes_sorted[len(nodes) - 1 - i]) for i in range(n_nodes)]
        
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



def test_minimum(G, name, sample_size=2, n_fails=0):
    """
    Function to test if the 'minimum' assortativity value achieved by the reverse HH can be improved upon. 
    Takes a Graph, its name, the sample size to test, number of consecutive failed attempts before quitting, 
    default is N^2 

    returns dataframe that is {'name': graph name,
                               'N': number of nodes
                               'r': The value of r that the graph has at teh start of the attempts to reduce r 
                                    on that row
                               'sample_size': The number of edges to try to rewire at once (default is 2) 
                               'can_rewire': The potential edges can be rewired
                               'fails': number of times the potential edges cannot be rewired, up to n_fails
                               'same_edges': Whether or not the edges that can be rewired are just the same edges 
                                             as the originals. There are a few combinations that may appear different
                                             but are the same. Need to check properly
                                } 
    """
    
    r_curr = nx.degree_assortativity_coefficient(G)
    j = 0
    def_row = {'name': name,
           'N': len(G.nodes()),
           'r': r_curr,
           'sample_size': sample_size, 
           'can_rewire': 0,
           'reduces': 0,
           'fails_to_reduce': 0,
           'duplicate_edges': 0, 
           'self_edges': 0,
           'existing_edges': 0,
           'failures': 0}
    
    row = def_row
    results = pd.DataFrame([row])

    while j < n_fails:
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
        n_nodes = int(len(nodes_sorted)/2)
        potential_edges = [(nodes_sorted[i], nodes_sorted[len(nodes) - 1 - i]) for i in range(n_nodes)]
        
        edges_to_add, row = check_new_edges(potential_edges, G, row)
        
        if len(edges_to_add) == len(potential_edges):
            row['can_rewire'] += 1
            G.remove_edges_from(edges_to_remove)
            G.add_edges_from(edges_to_add)
            r_new = nx.degree_assortativity_coefficient(G)
            if r_new < row['r']:
                row['reduces'] += 1
                print(f'manages to reduce from {row['r']} to {r_new}')
                results.loc[len(results)] = row

                j = 0
                row = def_row
                row['r'] = r_new
            else:
                row['fails_to_reduce'] += 1
                j += 1
        else:
            row['failures'] += 1
            j += 1

    results.loc[len(results)] = row

    return G, results

def test_maximum(G, name, sample_size=2, n_fails=0):
    """
    Function to test if the 'minimum' assortativity value achieved by the reverse HH can be improved upon. 
    Takes a Graph, its name, the sample size to test, number of consecutive failed attempts before quitting, 
    default is N^2 

    returns dataframe that is {'name': graph name,
                               'N': number of nodes
                               'r': The value of r that the graph has at teh start of the attempts to reduce r 
                                    on that row
                               'sample_size': The number of edges to try to rewire at once (default is 2) 
                               'can_rewire': The potential edges can be rewired
                               'fails': number of times the potential edges cannot be rewired, up to n_fails
                               'same_edges': Whether or not the edges that can be rewired are just the same edges 
                                             as the originals. There are a few combinations that may appear different
                                             but are the same. Need to check properly
                                } 
    """
    
    r_curr = nx.degree_assortativity_coefficient(G)
    j = 0
    def_row = {'name': name,
           'N': len(G.nodes()),
           'r': r_curr,
           'sample_size': sample_size, 
           'can_rewire': 0,
           'increases': 0,
           'fails_to_increase': 0,
           'duplicate_edges': 0, 
           'self_edges': 0,
           'existing_edges': 0,
           'failures': 0}
    
    row = def_row
    results = pd.DataFrame([row])

    while j < n_fails:
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
            row['can_rewire'] += 1
            G.remove_edges_from(edges_to_remove)
            G.add_edges_from(edges_to_add)
            r_new = nx.degree_assortativity_coefficient(G)
            if r_new > row['r']:
                row['increases'] += 1
                print(f'manages to increase from {row['r']} to {r_new}')
                results.loc[len(results)] = row

                j = 0
                row = def_row
                row['r'] = r_new
            else:
                row['fails_to_increase'] += 1
                j += 1
        else:
            row['failures'] += 1
            j += 1

    results.loc[len(results)] = row

    return G, results

