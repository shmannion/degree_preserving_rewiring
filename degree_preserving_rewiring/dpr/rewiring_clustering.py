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


