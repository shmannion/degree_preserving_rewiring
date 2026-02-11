
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
from .rewiring_helpers import degree_list, check_new_edges 

def havel_hakimi_positive(
    G: nx.Graph, 
    results, 
    name, 
    sample_size, 
    return_type, 
    max_time = 600):
    
    """
    removes every edge from the graph and adds them back ordered in such a way
    to maximise the assortativity.

    Parameters:
      G: nx.Graph
        graph to be rewired

      results: pandas.DataFrame
        results dataframe to be passed to function requiring the columns assigned
        in the rewiring function above

      sample_size: int
        number of edges to be rewired. Relevant only for passing the result of this 
        function to another
    
    Returns:
    --------
      G: nx.Graph
        rewired graph

      results: pandas.DataFrame
        results dataframe passed to the function with one row added per algorithm
        iteration
    """
    itr = 1
    before = degree_list(G)    
    alg_start = time.time()    
    edges_to_remove = list(G.edges())                
     
    #record the orginal degree of each node
    original_degree = {}
    remaining_degree = {}
    nodes = []
    for edge in edges_to_remove:
        for node in edge:
            if node not in nodes:
                nodes.append(node)
            original_degree[node] = G.degree(node)
            remaining_degree[node] = original_degree[node]
    #sort nodes in descending order of degree
    nodes = sorted(nodes, key=original_degree.get, reverse=True)
    target_nodes = nodes

    row = {'name' : name,
           'iteration' : itr, 
           'time' : 0, 
           'r' : 0,
           'target_r': 0,
           'sample_size': sample_size, 
           'edges_rewired': 0,
           'duplicate_edges': 0, 
           'self_edges': 0,
           'existing_edges': 0, 
           'preserved': True,
           'method': 'max',
           'summary': False}

    #dictionary in which to record the current neighbors of the nodes as we add edges 
    new_neighbors = {}
    for node in original_degree:
        new_neighbors[node] = set()


    for node in nodes:
        for target in target_nodes:
            if remaining_degree[node] > 0:
                if remaining_degree[target] > 0:
                    if node != target:
                        new_neighbors[node].add(target)
                        new_neighbors[target].add(node)
                        remaining_degree[node] -= 1
                        remaining_degree[target] -= 1


        target_nodes = sorted(target_nodes, key=remaining_degree.get, reverse=True)
    
    edges_to_add = []
    for node in new_neighbors:
        for target in new_neighbors[node]:
            edges_to_add.append([node, target])
    
    G.remove_edges_from(edges_to_remove)
    G.add_edges_from(edges_to_add)
    row['edges_rewired'] += len(edges_to_add)
    row['r'] += nx.degree_assortativity_coefficient(G)
    row['time'] += time.time() - alg_start
    after = degree_list(G)
    row['preserved'] = list(before) == list(after)
    results.loc[len(results)] = row
    
    edges = list(G.edges())
    
    #check to ensure that we have maintained the degree sequence
    success = True
    for node in original_degree:
        if G.degree(node) < original_degree[node]:
            success = False

    #if degree sequence has not been maintained, find the nodes with incorrect
    #degree and remove edges to rewire to them
    
    while success == False:
        itr += 1
        start = time.time()
        row = {'name': name,
               'iteration' : itr, 
               'time' : 0, 
               'r' : 0,
               'target_r': 0,
               'sample_size': sample_size, 
               'edges_rewired': 0,
               'duplicate_edges': 0, 
               'self_edges': 0,
               'existing_edges': 0, 
               'preserved': True,
               'method': 'max',
               'summary': False}

        affected_nodes = []
        missing_degree = {}
        
        for node in original_degree:
            if G.degree(node) != original_degree[node]:
                missing_degree[node] = original_degree[node] - G.degree(node)
                affected_nodes.append(node)
        
        for node in affected_nodes:
            available_edges = 0
            for target in affected_nodes:
                if target != node:
                    if target not in new_neighbors[node]:
                        available_edges += 1
    
        stubs1 = []
        stubs2 = []
        for node in missing_degree:
            j = 0
            while j < missing_degree[node]:
                stubs1.append(node)
                j += 1

        edges = list(G.edges())
        while len(stubs2) < len(stubs1):
            edge_to_remove = random.choice(edges)
            if edge_to_remove[0] not in stubs1:
                if edge_to_remove[1] not in stubs1:
                    if G.has_edge(edge_to_remove[0], edge_to_remove[1]) == True:
                        stubs2.append(edge_to_remove[0])
                        stubs2.append(edge_to_remove[1])
                        G.remove_edge(edge_to_remove[0], edge_to_remove[1])
            edges.remove(edge_to_remove)

        stubs1 = sorted(stubs1, key = original_degree.get, reverse=False)
        stubs2 = sorted(stubs2, key = original_degree.get, reverse=False)
        for u, v in zip(stubs1, stubs2):
            G.add_edge(u, v)
            row['edges_rewired'] += 1
        
        fails = 0
        for node in original_degree:
            if G.degree(node) != original_degree[node]:
                fails += 1
        
        if fails > 0:
            success = False
        else:
            success = True
    
        row['r'] += nx.degree_assortativity_coefficient(G)
        row['time'] += time.time() - start
        after = degree_list(G)
        row['preserved'] = list(before) == list(after)
        results.loc[len(results)] = row
        if return_type == 'full':
            results.loc[len(results)] = row

        if time.time() - alg_start > max_time:
            break

        results.loc[len(results)] = row
    return G

def havel_hakimi_negative(
    G: nx.Graph, 
    results, 
    name, 
    sample_size, 
    return_type, 
    max_time = 600):
    
    """
    removes every edge from the graph and adds them back ordered in such a way
    to minimise the assortativity.

    Parameters:
    -----------
      G: nx.Graph
        graph to be rewired

      results: pandas.DataFrame
        results dataframe to be passed to function requiring the columns assigned
        in the rewiring function above

      sample_size: int
        number of edges to be rewired. Relevant only for passing the result of this 
        function to another

    Returns:
    --------
      G: nx.Graph
        rewired graph

      results: pandas.DataFrame
        results dataframe passed to the function with one row added per algorithm
        iteration
    """
    before = degree_list(G)    
    alg_start = time.time()    
    edges_to_remove = list(G.edges())                
    itr = 1
    #record the orginal degree of each node
    original_degree = {}
    remaining_degree = {}
    nodes = []
    for edge in edges_to_remove:
        for node in edge:
            if node not in nodes:
                nodes.append(node)
            original_degree[node] = G.degree(node)
            remaining_degree[node] = original_degree[node]
    

    #sort nodes in descending order of degree
    nodes = sorted(nodes, key=original_degree.get, reverse=False)
    target_nodes = list(reversed(nodes))
    row = {'name': name,
           'iteration' : itr, 
           'time' : 0, 
           'r' : 0,
           'target_r': 0,
           'sample_size': sample_size, 
           'edges_rewired': 0,
           'duplicate_edges': 0, 
           'self_edges': 0,
           'existing_edges': 0, 
           'preserved': True,
           'method': 'max',
           'summary': False}

    #dictionary in which to record the new neighbours we are adding 
    new_neighbors = {}
    for node in original_degree:
        new_neighbors[node] = set() 

    edges_to_add = []
    
    for node in nodes:
        for target in target_nodes:
            if remaining_degree[node] > 0:
                if remaining_degree[target] > 0:
                    if node != target:
                        new_neighbors[node].add(target)
                        new_neighbors[target].add(node)
                        remaining_degree[node] -= 1
                        remaining_degree[target] -= 1


        target_nodes = sorted(target_nodes, key=remaining_degree.get, reverse=True)
    
    edges_to_add = []
    for node in new_neighbors:
        for target in new_neighbors[node]:
            edge = [node, target]
            edges_to_add.append([node, target])
    
    G.remove_edges_from(edges_to_remove)
    G.add_edges_from(edges_to_add)
    row['edges_rewired'] += len(edges_to_add) 
    row['r'] += nx.degree_assortativity_coefficient(G)
    row['time'] += time.time() - alg_start
    after = degree_list(G)
    row['preserved'] = list(before) == list(after)
    results.loc[len(results)] = row
    
    edges = list(G.edges())
    
    success = True
    for node in original_degree:
        if G.degree(node) < original_degree[node]:
            success = False

    while success == False:
        itr += 1
        start = time.time()
        row = {'name': name,
               'iteration' : itr, 
               'time' : 0, 
               'r' : 0,
               'target_r': 0,
               'sample_size': sample_size, 
               'edges_rewired': 0,
               'duplicate_edges': 0, 
               'self_edges': 0,
               'existing_edges': 0, 
               'preserved': True,
               'method': 'max',
               'summary': False}

        affected_nodes = []
        missing_degree = {}
        for node in original_degree:
            if G.degree(node) != original_degree[node]:
                missing_degree[node] = original_degree[node] - G.degree(node)
                affected_nodes.append(node)
   
        for node in affected_nodes:
            available_edges = 0
            for target in affected_nodes:
                if target != node:
                    if target not in new_neighbors[node]:
                        available_edges += 1
    
        stubs1 = []
        stubs2 = []
        for node in missing_degree:
            j = 0
            while j < missing_degree[node]:
                stubs1.append(node)
                j += 1
       
        edges = list(G.edges())
        while len(stubs2) < len(stubs1):
            edge_to_remove = random.choice(edges)
            if edge_to_remove[0] not in stubs1:
                if edge_to_remove[1] not in stubs1:
                    if G.has_edge(edge_to_remove[0], edge_to_remove[1]) == True:
                        stubs2.append(edge_to_remove[0])
                        stubs2.append(edge_to_remove[1])
                        G.remove_edge(edge_to_remove[0], edge_to_remove[1])
            edges.remove(edge_to_remove)

        stubs1 = sorted(stubs1, key = original_degree.get, reverse=False)
        stubs2 = sorted(stubs2, key = original_degree.get, reverse=True)
        for u, v in zip(stubs1, stubs2):
            G.add_edge(u, v)
            row['edges_rewired'] += 1 
        
        fails = 0
        for node in original_degree:
            if G.degree(node) != original_degree[node]:
                fails += 1
        
        if fails > 0:
            success = False
        else:
            success = True
         
        row['r'] += nx.degree_assortativity_coefficient(G)
        row['time'] += time.time() - start
        after = degree_list(G)
        row['preserved'] = list(before) == list(after)
        if return_type == 'full':
            results.loc[len(results)] = row

        if time.time() - alg_start > max_time:
            break
    
        results.loc[len(results)] = row
    
    return G

