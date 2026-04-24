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
from collections import defaultdict
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
        return G, summarised_results
    
    else:
        return G, results




def positively_rewire(
    G: nx.Graph, 
    target_assortativity, 
    name, 
    results, 
    sample_size = 2, 
    timed = True, 
    time_limit=600,
    property_checks=False):
    
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


def reduce_clustering(
    G: nx.Graph,
    name,
    results,
    target_clustering=None,
    max_iterations=None,
    max_consecutive_failures=10000,
    timed=False,
    time_limit=600,
    log_failures=False):
    """
    Reduces the clustering coefficient of G using same-degree neighbor swaps.

    A swap picks two nodes u, v with identical degree and swaps one neighbour
    each: edges (u, b) and (v, y) become (v, b) and (u, y). Because u and v
    have the same degree, every edge endpoint keeps the same degree-pair, so
    assortativity is preserved exactly. Only swaps whose local triangle delta
    is strictly negative are accepted.

    Parameters
    ----------
    G : nx.Graph
        Graph to rewire (modified in place).
    name : str
        Name recorded in the results DataFrame.
    results : pandas.DataFrame
        Results DataFrame; one row appended per accepted swap (and per failed
        attempt if log_failures is True).
    target_clustering : float, optional
        Stop once nx.transitivity(G) <= target_clustering. Disabled if None.
    max_iterations : int, optional
        Stop after this many iterations (attempted + accepted). Disabled if None.
    max_consecutive_failures : int
        Stop after this many consecutive failed attempts (convergence heuristic).
    timed : bool
        If True, stop after time_limit seconds.
    time_limit : float
        Time limit in seconds.
    log_failures : bool
        If True, log every attempt; if False, log only accepted swaps.

    Returns
    -------
    G : nx.Graph
        Rewired graph.
    """
    alg_start = time.time()
    itr = 0
    consecutive_failures = 0

    degree_classes = defaultdict(list)
    for node in G.nodes():
        degree_classes[G.degree(node)].append(node)
    usable_degrees = [d for d, lst in degree_classes.items() if len(lst) >= 2]

    if not usable_degrees:
        return G

    # Same-degree neighbour swaps preserve degrees exactly, so assortativity
    # and each node's 1/C(d_v,2) weight are invariants.
    r_invariant = nx.degree_assortativity_coefficient(G)

    n_nodes = G.number_of_nodes()
    degrees = dict(G.degree())
    # inv_weight[v] = 2 / (N * d_v * (d_v - 1)), contribution of Δt_v to ΔC_avg
    inv_weight = {}
    for node, d in degrees.items():
        inv_weight[node] = (2.0 / (n_nodes * d * (d - 1))) if d >= 2 else 0.0

    # Per-node triangle counts; maintained incrementally thereafter.
    t = nx.triangles(G)
    C_avg = sum(t[v] * inv_weight[v] for v in t)

    # Cache neighbour sets; update only the four touched nodes per accepted swap.
    neighbours = {n: set(G.neighbors(n)) for n in G.nodes()}

    while True:
        loop_start = time.time()
        if max_iterations is not None and itr >= max_iterations:
            print(f'exiting due to max iterations reached, took {time.time() - alg_start}')
            break
        if consecutive_failures >= max_consecutive_failures:
            print(f'exiting due to max failures reached, took {time.time() - alg_start}')
            break
        if timed and (time.time() - alg_start) > time_limit:
            print(f'exiting due to max time reached, took {time.time() - alg_start}')
            break
        if target_clustering is not None and C_avg <= target_clustering:
            print(f'exiting due to target reached, took {time.time() - alg_start}')
            break

        itr += 1
        accepted = False
        reason = None

        degree = random.choice(usable_degrees)
        u, v = random.sample(degree_classes[degree], 2)

        N_u = neighbours[u]
        N_v = neighbours[v]
        if not N_u or not N_v:
            consecutive_failures += 1
        else:
            b = random.choice(tuple(N_u))
            y = random.choice(tuple(N_v))

            if b == y or b == v or y == u:
                reason = 'self_edges'
                consecutive_failures += 1
            elif b in N_v or y in N_u:
                reason = 'existing_edges'
                consecutive_failures += 1
            else:
                N_b = neighbours[b]
                N_y = neighbours[y]

                # Enumerate affected triangles as sets of third-vertices w.
                # Destroyed: (u,b,w) for w ∈ W_ub, and (v,y,w) for w ∈ W_vy.
                W_ub = N_u & N_b
                W_vy = N_v & N_y
                # Created: (v,b,w) and (u,y,w). Pre-swap N(v)∩N(b) may include
                # u (if u∈N_v) or y (if y∈N_b) — neither forms a post-swap
                # triangle because (u,b) and (v,y) are gone. Same for N(u)∩N(y)
                # with v and b. Exclude them.
                W_vb = (N_v & N_b) - {u, y}
                W_uy = (N_u & N_y) - {v, b}

                # ΔC_avg = sum_v (Δt_v) * inv_weight[v].
                # Each triangle contributes -1/+1 to all three of its vertices.
                dC = 0.0
                if W_ub:
                    s_w = sum(inv_weight[w] for w in W_ub)
                    dC -= len(W_ub) * (inv_weight[u] + inv_weight[b]) + s_w
                if W_vy:
                    s_w = sum(inv_weight[w] for w in W_vy)
                    dC -= len(W_vy) * (inv_weight[v] + inv_weight[y]) + s_w
                if W_vb:
                    s_w = sum(inv_weight[w] for w in W_vb)
                    dC += len(W_vb) * (inv_weight[v] + inv_weight[b]) + s_w
                if W_uy:
                    s_w = sum(inv_weight[w] for w in W_uy)
                    dC += len(W_uy) * (inv_weight[u] + inv_weight[y]) + s_w

                if dC < 0:
                    G.remove_edge(u, b)
                    G.remove_edge(v, y)
                    G.add_edge(v, b)
                    G.add_edge(u, y)
                    N_u.discard(b); N_u.add(y)
                    N_v.discard(y); N_v.add(b)
                    N_b.discard(u); N_b.add(v)
                    N_y.discard(v); N_y.add(u)

                    # Update per-node triangle counts.
                    for w in W_ub:
                        t[u] -= 1; t[b] -= 1; t[w] -= 1
                    for w in W_vy:
                        t[v] -= 1; t[y] -= 1; t[w] -= 1
                    for w in W_vb:
                        t[v] += 1; t[b] += 1; t[w] += 1
                    for w in W_uy:
                        t[u] += 1; t[y] += 1; t[w] += 1

                    C_avg += dC
                    accepted = True
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1

        if accepted or log_failures:
            row = {'name': name,
                   'iteration': itr,
                   'time': time.time() - loop_start,
                   'r': r_invariant,
                   'target_r': 0,
                   'sample_size': 2,
                   'edges_rewired': 2 if accepted else 0,
                   'duplicate_edges': 0,
                   'self_edges': 1 if reason == 'self_edges' else 0,
                   'existing_edges': 1 if reason == 'existing_edges' else 0,
                   'preserved': True,
                   'method': 'reduce_clustering',
                   'summary': False}
            results.loc[len(results)] = row

    return G


def reduce_clustering_unconstrained(
    G: nx.Graph,
    name,
    results,
    target_clustering=None,
    max_iterations=None,
    max_consecutive_failures=10000,
    timed=False,
    time_limit=600,
    log_failures=False):
    """
    Reduces the clustering coefficient of G via double-edge swaps that preserve
    the degree sequence but NOT degree assortativity. Intended as an empirical
    lower-bound estimator for the minimum clustering achievable at a given
    degree sequence, unconstrained by r.

    Two random edges (u, b) and (v, y) are selected; endpoint orientation is
    randomised so both reachable swap outcomes are equally likely. The swap
    (u, b), (v, y) -> (v, b), (u, y) is accepted iff the exact change in
    average clustering is strictly negative.

    Parameters match reduce_clustering. See its docstring for column meanings.

    Notes
    -----
    Assortativity changes during the run. The 'r' column records the value at
    function entry only; call nx.degree_assortativity_coefficient(G) on the
    returned graph for the post-run value.
    """
    alg_start = time.time()
    itr = 0
    consecutive_failures = 0

    n_nodes = G.number_of_nodes()
    degrees = dict(G.degree())
    inv_weight = {}
    for node, d in degrees.items():
        inv_weight[node] = (2.0 / (n_nodes * d * (d - 1))) if d >= 2 else 0.0

    t = nx.triangles(G)
    C_avg = sum(t[v] * inv_weight[v] for v in t)

    neighbours = {n: set(G.neighbors(n)) for n in G.nodes()}
    r_start = nx.degree_assortativity_coefficient(G)

    # Edge list + index map: O(1) uniform sampling and O(1) swap-pop removal.
    def _canon(a, b):
        return (a, b) if a <= b else (b, a)

    edge_list = [_canon(u, v) for u, v in G.edges()]
    edge_index = {e: i for i, e in enumerate(edge_list)}

    def _remove_edge(e):
        i = edge_index.pop(e)
        last = edge_list[-1]
        if i != len(edge_list) - 1:
            edge_list[i] = last
            edge_index[last] = i
        edge_list.pop()

    def _add_edge(e):
        edge_index[e] = len(edge_list)
        edge_list.append(e)

    while True:
        loop_start = time.time()
        if max_iterations is not None and itr >= max_iterations:
            print(f'exiting due to max iterations reached, took {time.time() - alg_start}')
            break
        if consecutive_failures >= max_consecutive_failures:
            print(f'exiting due to max failures reached, took {time.time() - alg_start}')
            break
        if timed and (time.time() - alg_start) > time_limit:
            print(f'exiting due to max time reached, took {time.time() - alg_start}')
            break
        if target_clustering is not None and C_avg <= target_clustering:
            print(f'exiting due to target reached, took {time.time() - alg_start}')
            break

        itr += 1
        accepted = False
        reason = None

        e1, e2 = random.sample(edge_list, 2)
        u, b = e1
        v, y = e2
        if random.random() < 0.5:
            u, b = b, u
        if random.random() < 0.5:
            v, y = y, v

        if u == v or u == y or b == v or b == y:
            reason = 'self_edges'
            consecutive_failures += 1
        else:
            N_u = neighbours[u]
            N_v = neighbours[v]
            N_b = neighbours[b]
            N_y = neighbours[y]

            if b in N_v or y in N_u:
                reason = 'existing_edges'
                consecutive_failures += 1
            else:
                W_ub = N_u & N_b
                W_vy = N_v & N_y
                W_vb = (N_v & N_b) - {u, y}
                W_uy = (N_u & N_y) - {v, b}

                dC = 0.0
                if W_ub:
                    s_w = sum(inv_weight[w] for w in W_ub)
                    dC -= len(W_ub) * (inv_weight[u] + inv_weight[b]) + s_w
                if W_vy:
                    s_w = sum(inv_weight[w] for w in W_vy)
                    dC -= len(W_vy) * (inv_weight[v] + inv_weight[y]) + s_w
                if W_vb:
                    s_w = sum(inv_weight[w] for w in W_vb)
                    dC += len(W_vb) * (inv_weight[v] + inv_weight[b]) + s_w
                if W_uy:
                    s_w = sum(inv_weight[w] for w in W_uy)
                    dC += len(W_uy) * (inv_weight[u] + inv_weight[y]) + s_w

                if dC < 0:
                    G.remove_edge(u, b)
                    G.remove_edge(v, y)
                    G.add_edge(v, b)
                    G.add_edge(u, y)
                    N_u.discard(b); N_u.add(y)
                    N_v.discard(y); N_v.add(b)
                    N_b.discard(u); N_b.add(v)
                    N_y.discard(v); N_y.add(u)

                    for w in W_ub:
                        t[u] -= 1; t[b] -= 1; t[w] -= 1
                    for w in W_vy:
                        t[v] -= 1; t[y] -= 1; t[w] -= 1
                    for w in W_vb:
                        t[v] += 1; t[b] += 1; t[w] += 1
                    for w in W_uy:
                        t[u] += 1; t[y] += 1; t[w] += 1

                    _remove_edge(e1)
                    _remove_edge(e2)
                    _add_edge(_canon(v, b))
                    _add_edge(_canon(u, y))

                    C_avg += dC
                    accepted = True
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1

        if accepted or log_failures:
            row = {'name': name,
                   'iteration': itr,
                   'time': time.time() - loop_start,
                   'r': r_start,
                   'target_r': 0,
                   'sample_size': 2,
                   'edges_rewired': 2 if accepted else 0,
                   'duplicate_edges': 0,
                   'self_edges': 1 if reason == 'self_edges' else 0,
                   'existing_edges': 1 if reason == 'existing_edges' else 0,
                   'preserved': True,
                   'method': 'reduce_clustering_unconstrained',
                   'summary': False}
            results.loc[len(results)] = row

    return G


def connect_components(
    G: nx.Graph,
    name,
    results,
    max_attempts=50):
    """
    Merges disconnected components of G via random inter-component double-edge
    swaps. Preserves the degree sequence; does NOT preserve assortativity.
    Intended as a fast reconnection step when small shifts in r are acceptable.

    At each iteration a random edge is drawn from the smallest non-trivial
    component and from the current largest component, and a double-edge swap
    (a1,a2),(b1,b2) -> (a1,b1),(a2,b2) merges them. If BOTH chosen edges are
    bridges in their components the swap splits the graph rather than merging,
    so each attempt is verified with has_path(a1, a2) and reverted on failure.
    Up to max_attempts attempts per merge before giving up.

    Isolated (degree-0) nodes cannot be reached by edge swap and are left as
    separate components.

    Parameters
    ----------
    G : nx.Graph
        Graph to be merged (modified in place).
    name : str
        Name recorded in the results DataFrame.
    results : pandas.DataFrame
        One row appended per executed merge.
    max_attempts : int
        Maximum edge-pair attempts per merge. Default 50.

    Returns
    -------
    G : nx.Graph
        Graph with one non-trivial connected component (assuming all merges
        succeeded).
    """
    itr = 0

    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    if isolated:
        print(f'warning: {len(isolated)} isolated node(s) of degree 0 cannot '
              f'be merged by edge swap and will remain as separate components')

    components = [list(c) for c in nx.connected_components(G) if len(c) > 1]
    r_start = nx.degree_assortativity_coefficient(G)
    print(f'starting connect_components: {len(components)} non-trivial components, '
          f'r={r_start:.4f}')

    components.sort(key=len)

    while len(components) > 1:
        loop_start = time.time()
        itr += 1

        small = components[0]
        main = components[-1]

        merged = False
        for _ in range(max_attempts):
            a1 = random.choice(main)
            a2 = random.choice(list(G.neighbors(a1)))
            b1 = random.choice(small)
            b2 = random.choice(list(G.neighbors(b1)))

            G.remove_edge(a1, a2)
            G.remove_edge(b1, b2)
            G.add_edge(a1, b1)
            G.add_edge(a2, b2)

            # The swap merges iff at least one of (a1,a2) and (b1,b2) is not
            # a bridge. When both are bridges the graph splits instead:
            # verify by checking connectivity of the two main endpoints.
            if nx.has_path(G, a1, a2):
                merged = True
                break

            G.remove_edge(a1, b1)
            G.remove_edge(a2, b2)
            G.add_edge(a1, a2)
            G.add_edge(b1, b2)

        if not merged:
            print(f'warning: could not merge on iteration {itr} after '
                  f'{max_attempts} attempts (both components may be tree-like); '
                  f'{len(components)} components left unmerged')
            break

        # small absorbed into main; main can only grow, so it stays at [-1].
        main.extend(small)
        components.pop(0)

        row = {'name': name,
               'iteration': itr,
               'time': time.time() - loop_start,
               'r': 0,
               'target_r': 0,
               'sample_size': 2,
               'edges_rewired': 2,
               'duplicate_edges': 0,
               'self_edges': 0,
               'existing_edges': 0,
               'preserved': True,
               'method': 'connect_components',
               'summary': False}
        results.loc[len(results)] = row


    print(f'done: r={nx.degree_assortativity_coefficient(G):.4f}')
    return G

