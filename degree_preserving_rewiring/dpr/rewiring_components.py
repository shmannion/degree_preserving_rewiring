# -*- coding: utf-8 -*-
"""
Created on Tues Apr 28 2026 

@author: shane mannion
"""

import networkx as nx
import numpy as np
import pandas as pd
import time
import random
from collections import defaultdict
from .havel_hakimi import havel_hakimi_positive, havel_hakimi_negative
from .rewiring_helpers import degree_list, check_new_edges, test_sample_sizes

def connect_components(
    G: nx.Graph,
    name,
    results,
    max_attempts=50):
    """
    Merges disconnected components of G via random inter-component double-edge
    swaps. Preserves the degree sequence; does NOT preserve assortativity.
    Intended as a fast reconnection step when small shifts in r are acceptable.

    At each iteration a random edge is drawn from the smallest component and from the 
    current largest component, and a double-edge swap (a1,a2),(b1,b2) -> (a1,b1),(a2,b2) 
    merges them. 

    If both chosen edges are such that removing them creates new components, 
    the swap splits the graph rather than merging, so each attempt is 
    verified with has_path(a1, a2) and reverted on failure.
    
    Attempts to connect graph max_attempts times per merge before giving up.

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

