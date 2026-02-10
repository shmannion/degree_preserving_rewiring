# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:05:08 2023

@author: shane
"""

import numpy as np
from .MLE.MLE_functions import MLE, degree_list
import networkx as nx

def generate_graph(distribution, target_mean, size, params='default'):
    """
    Function to call chosen graph generation function from this file

    Parameters:
    ----------
        distribution: str
            desired degree distribution of graph

        target_mean: double
            the desired average degree of the graph

        size:
            the number of nodes desired in the graph

    Returns:
    ---------
        G: networkx.graph
            graph with chosen degree distribution
    """
    if params == 'default':
        if distribution == 'weibull':
            x, G = generate_weibull(target_mean, size)
        if distribution == 'lognormal':
            x, G = generate_lognormal(target_mean, size)
        if distribution == 'exponential':
            x, G = generate_exponential(target_mean, size)
    else:
        if distribution == 'weibull':
            x, G = generate_weibull(target_mean, size, params)
        if distribution == 'lognormal':
            x, G = generate_lognormal(target_mean, size, params)
        if distribution == 'exponential':
            x, G = generate_exponential(target_mean, size, params)
    

    return G

def generate_weibull(target_mean, size, params = [2.1, 0.48]):
    """
    Function to generate graph with a weibull distribution
    set p1 = 2.1, p2=0.48 for mean degree of 5

    Parameters
    ----------
    target_mean : float
        desired mean degree.
    p1 : float
        first distribution parameter.
    p2 : float
        second distribution parameter.
    size : int
        graph size.

    Returns
    -------
    x : list
        degree list.
    G : nx.Graph
        Graph with desired distribution.

    """
    inf = np.arange(1000)
    Input = [i for i in range(1,5000)]
    k_min = 1
    C = 1

    dist = None
    sum1 = np.array([np.sum((((j+inf)/params[0])**(params[1]-1))*np.exp(-(((j+inf)/params[0])**params[1]))) for j in Input])
    inf_sum = np.sum((((inf + k_min)/params[0])**(params[1]-1)*np.exp(-1*((inf + k_min)/params[0])**params[1])))
    y = C*sum1/inf_sum

    cdf = 1 - y


    while dist != 'Weibull':
        pvals = np.random.uniform(0,1, size)

        x = []
        for val in pvals:
            for i in range(len(cdf) - 1):
                if val >= cdf[i]:
                    if val < cdf[i + 1]:
                        x.append(i + 1)
        
        try:
            MG = nx.configuration_model(x)
        except nx.NetworkXError:
            x[0] += 1
            MG = nx.configuration_model(x)
    
    
        G = nx.Graph(MG)
        G.remove_edges_from(nx.selfloop_edges(G))
        try:
            dist = MLE(degree_list(G))[1]
        except ValueError:
            dist = None
    return x, G        

def generate_lognormal(target_mean, size, params = [1.4, 0.6]):
    """
    Function to generate graph with a lognormal distribution
    set p1 = 1.4, p2 = 0.6 for mean degree of 5

    Parameters
    ----------
    target_mean : float
        desired mean degree.
    p1 : float
        first distribution parameter.
    p2 : float
        second distribution parameter.
    size : int
        graph size.

    Returns
    -------
    x : list
        degree list.
    G : nx.Graph
        Graph with desired distribution.

    """
    inf = np.arange(1000)
    Input = [i for i in range(1,5000)]
    k_min = 1
    C = 1

    dist = None
    sum1 = np.array([np.sum( (1.0/(j+inf))*np.exp(-((np.log(j+inf)-params[0])**2)/(2*(params[1]**2)))) for j in Input])
    inf_sum = np.sum( (1.0/(inf+k_min)) * np.exp(-((np.log(inf+k_min)-params[0])**2)/(2*params[1]**2) ) )
    y = C*sum1/(inf_sum)  

    cdf = 1 - y


    while dist != 'Lognormal':
        pvals = np.random.uniform(0,1, size)

        x = []
        for val in pvals:
            for i in range(len(cdf) - 1):
                if val >= cdf[i]:
                    if val < cdf[i + 1]:
                        x.append(i + 1)
        
        try:
            MG = nx.configuration_model(x)
        except nx.NetworkXError:
            x[0] += 1
            MG = nx.configuration_model(x)
    
    
        G = nx.Graph(MG)
        G.remove_edges_from(nx.selfloop_edges(G))
        try:
            dist = MLE(degree_list(G))[1]
        except ValueError:
            dist = None
    return x, G        



def generate_exponential(target_mean, size, params = [4.5]):
    """
    Function to generate graph with an exponential distribution
    set p1 = 4.5, for mean degree of 5

    Parameters
    ----------
    target_mean : float
        desired mean degree.
    p1 : float
        first distribution parameter.
    p2 : float
        second distribution parameter.
    size : int
        graph size.

    Returns
    -------
    x : list
        degree list.
    G : nx.Graph
        Graph with desired distribution.

    """
    Input = np.arange(1,5000)
    k_min = 1
    C = 1

    dist = None
    y = C*np.exp((-1/params[0])*(Input-k_min))
        
    
    cdf = 1 - y


    while dist != 'Exponential':
        pvals = np.random.uniform(0,1, size)

        x = []
        for val in pvals:
            for i in range(len(cdf) - 1):
                if val >= cdf[i]:
                    if val < cdf[i + 1]:
                        x.append(i + 1)
        
        try:
            MG = nx.configuration_model(x)
        except nx.NetworkXError:
            x[0] += 1
            MG = nx.configuration_model(x)
    
    
        G = nx.Graph(MG)
        G.remove_edges_from(nx.selfloop_edges(G))
        try:
            dist = MLE(degree_list(G))[1]
        except ValueError:
            dist = None
    return x, G        

