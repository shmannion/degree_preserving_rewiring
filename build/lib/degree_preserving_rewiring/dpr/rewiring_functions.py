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
from .havel_hakimi import havel_hakimi_positive, havel_hakimi_negative
from .rewiring_helpers import degree_list, check_new_edges, test_sample_sizes

def rewire(
    G, 
    target_assortativity, 
    name, 
    sample_size = 2, 
    timed = False, 
    time_limit=600, 
    method='new', 
    return_type = 'full'):
    """
    Parameters
    ----------

    G : networkx.Graph
        graph to be reiwired
    target_assortativity : float in range [-1, 1]
        desired value for assortativity
    name: str
        name to appear in results data set
    sample_size : int
        number of edges to rewire at each iteration
    timed : bool
        whether or not to impose a maximum time on the algorithm
    time_limit : float
        time limit if the algorithm is timed
    method : string
        can be 'new', 'old' or 'max'
            new: method described in paper [ADD REF WHEN AVAILABLE]

            old: original algorithm from Van Meighem et al. (2010)

            max: only step one of new version
    return_type: string
        can be 'full' or 'summarised'
            'full' : returns detailed results at each algorithm iteration

            'summarised': returns only total time taken, total iterations, etc.

    Returns:
    --------
    G : networkx.Graph
        rewired graph, rewiring done in place

    results : pandas.DataFrame()
        dataframe with all necessary info to plot results
        columns:
        iteration : number of loops completed so far (unsuccessful loops included)
        time : time taken for the current loop
        r : assortativity of the graph at the END of the current iteration
        sample_size : number of edges being selected at each iteration
                      N.B. The first loop will have a sample size = to the number of edges
                      but the row will be given the sample_size value of the succeeding rows
                      to allow for easy grouping
        edges_selected : cumulative number of edges sampled (unnsuccessful loops included)
        edges_rewired : cumulative number of edges rewired 
        duplicate_edges : The number of duplicate edges in the list of potential edges (one edge appearing twice = 1 here)
        self_edges : The number of self edges in the list of potential edges
        existing_edges : The number of edges in the list of potential edges that already exist in the graph
        preserved : If the degree_list has been preserved (only present in first and last rows)
        method : The method applied. 0 = none (for info about the starting values)
                                     1 = new method, rewiring_full phase
                                     2 = new method, second phase
        summary : Whether or not the row is a summary of the entire rewiring process for a graph
    

    """
    b_start = time.time()
    first_row = {'name':name,
                 'iteration': 0, 
                 'time': 0, 
                 'r': nx.degree_assortativity_coefficient(G),
                 'target_r': target_assortativity, 
                 'sample_size': sample_size, 
                 'edges_rewired': 0,
                 'duplicate_edges': 0, 
                 'self_edges': 0,
                 'existing_edges': 0, 
                 'preserved': True,
                 'method': method,
                 'summary':False}
    
    results = pd.DataFrame([first_row])

    before = degree_list(G)
    if nx.degree_assortativity_coefficient(G) < target_assortativity:
      if method == 'new':
        G = havel_hakimi_positive(G, results, name, sample_size, return_type)
        G = negatively_rewire(G, target_assortativity, name, results, sample_size, timed, time_limit)
      if method == 'original':
        G = positively_rewire(G, target_assortativity, name, results, sample_size, timed, time_limit)
      if method == 'max':
        G = havel_hakimi_positive(G, results, name, sample_size, return_type)

    else:
      if method == 'new':
        a_start = time.time()
        G = havel_hakimi_negative(G, results, name, sample_size, return_type)
        G = positively_rewire(G, target_assortativity, name, results, sample_size, timed, time_limit)
        a_end = time.time()
      if method == 'original':
        G = negatively_rewire(G, target_assortativity, name, results, sample_size, timed, time_limit)
      if method == 'max':
        G = havel_hakimi_negative(G, results, name, sample_size, return_type)

    after = degree_list(G)
    #we now have a dataframe of all of our relevant results
    summary_row = {'name': results.loc[results.index[-1], 'name'],
                   'iteration': results.loc[results.index[-1], 'iteration'], 
                   'time': results['time'].sum(), 
                   'r': nx.degree_assortativity_coefficient(G),
                   'target_r': target_assortativity, 
                   'sample_size': sample_size, 
                   'edges_rewired': results['edges_rewired'].sum(),
                   'duplicate_edges': results['duplicate_edges'].sum(), 
                   'self_edges': results['self_edges'].sum(),
                   'existing_edges': results['existing_edges'].sum(), 
                   'preserved': list(before) == list(after),
                   'method': 0,
                   'summary': True}

    if method == 'new':
        summary_row['method'] = 'new'
    if method == 'original':
        summary_row['method'] = 'original'
    if method == 'max':
        summary_row['method'] = 'max'

    results.loc[len(results)] = summary_row
    b_end = time.time()
    if return_type == 'summary':
        summarised_results = results.loc[(results['summary']==1)]
        return summarised_results
    
    else:
        return G, results




def positively_rewire(
    G: nx.Graph, 
    target_assortativity, 
    name, 
    results, 
    sample_size = 2, 
    timed = True, 
    time_limit=600):
    
    """
    Function for fine tuning the assortativity value of a graph.
    
    Parameters
    ----------
    G: nx.Graph
      Graph to be rewired

    target_assortativity: double
      desired assortativity value

    results: pandas.DataFrame
      DataFrame to be added to. One row per iteration. Must have columns as in 
      rewire function

    sample_size: int
      number of edges to be rewired per iteration. The default is 2

    timed: bool
      whether or not to stop the algorithm after a certain amount of time. The default 
      is True

    time_limit: double
      time after which to stop iterating. The default is 600 seconds.

    Returns
    -------
    G: nx.Graph
      rewired graph

    results: pandas.DataFrame
      DataFrame of results, one line per iteration
    """

    alg_start = time.time()
    itr = 1
    r = nx.degree_assortativity_coefficient(G)
    while r < target_assortativity:
        loop_start = time.time()
        itr += 1
        #define dictionary to track relevant info for each loop
        row = {'name': name,
               'iteration' : itr, 
               'time' : 0, 
               'r' : 0,
               'target_r': target_assortativity,
               'sample_size': sample_size, 
               'edges_rewired': 0,
               'duplicate_edges': 0, 
               'self_edges': 0,
               'existing_edges': 0, 
               'preserved': True,
               'method': 'new',
               'summary': False}

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
        G.remove_edges_from(edges_to_remove)
        edges_to_add, row = check_new_edges(potential_edges, G, row)
                
        if len(edges_to_add) == sample_size:
            G.add_edges_from(edges_to_add)
            row['edges_rewired'] += sample_size
        else:
            G.add_edges_from(edges_to_remove)

        r = nx.degree_assortativity_coefficient(G)
        row['r'] = r
        row['time'] += time.time() - loop_start
        results.loc[len(results)] = row

        time_elapsed = time.time() - alg_start
        
        if timed == True:
            if time_elapsed > time_limit:
                return G

    return G


def negatively_rewire(
    G: nx.Graph, 
    target_assortativity, 
    name, 
    results, 
    sample_size = 2, 
    timed = False, 
    time_limit=600):
    
    """
    Function for fine tuning the assortativity value of a graph.
    
    Parameters
    ----------
    G: nx.Graph
      Graph to be rewired

    target_assortativity: double
      desired assortativity value

    results: pandas.DataFrame
      DataFrame to be added to. One row per iteration. Must have columns as in rewire 
      function

    sample_size: int
      number of edges to be rewired per iteration. The default is 2

    timed: bool
      whether or not to stop the algorithm after a certain amount of time. The default 
      is True

    time_limit: double
      time after which to stop iterating. The default is 600 seconds.

    Returns
    -------
    G: nx.Graph
      rewired graph

    results: pandas.DataFrame
      DataFrame of results, one line per iteration
    """
    
    alg_start = time.time()
    itr = 0
    r = nx.degree_assortativity_coefficient(G)
    while r > target_assortativity:
        loop_start = time.time()
        itr += 1
        #define dictionary to track relevant info for each loop
        row = {'name' : name,
               'iteration' : itr, 
               'time' : 0, 
               'r' : 0,
               'target_r': target_assortativity,
               'sample_size': sample_size, 
               'edges_rewired': 0,
               'duplicate_edges': 0, 
               'self_edges': 0,
               'existing_edges': 0, 
               'preserved': True,
               'method': 'new',
               'summary': False}

        edges = list(G.edges())                
        edges_to_remove = random.sample(edges, sample_size)
        deg_dict = {}
        nodes = []
        for edge in edges_to_remove:
            for node in edge:
                nodes.append(node)
                deg_dict[node] = G.degree(node)
    
        nodes_sorted = sorted(nodes, key = deg_dict.get)
        n_nodes = int(len(nodes_sorted)/2)
        
        potential_edges = [(nodes_sorted[i], nodes_sorted[len(nodes) - 1 - i]) for i in range(n_nodes)]
        G.remove_edges_from(edges_to_remove)
        edges_to_add, row = check_new_edges(potential_edges, G, row)
        
        if len(edges_to_add) == len(potential_edges):
            G.add_edges_from(edges_to_add)
            row['edges_rewired'] += sample_size
        else:
            G.add_edges_from(edges_to_remove)

        r = nx.degree_assortativity_coefficient(G)
        row['time'] += time.time() - loop_start
        row['r'] = r
        results.loc[len(results)] = row
        time_elapsed = time.time() - alg_start
        
        if timed == True:
            if time_elapsed > time_limit:
                return G

    return G

