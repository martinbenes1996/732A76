# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 08:33:52 2020

@author: martin
"""

import numpy as np
from ortools.constraint_solver import pywrapcp,routing_enums_pb2
import pandas as pd
from scipy.spatial.distance import cdist, euclidean
from seriate import seriate

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9,7)
plt.rcParams['font.size'] = 7

import src
from tools import great_circle

# brute force
def _czekanowski_brute_seriate(D):
    
    def Um(M):
        v,D = [],[]
        for i in range(M.shape[0]):
            for j in range(i+1, M.shape[1]):
                v.append(M[i,j])
                D.append(abs(i - j))
        v,D = np.array(v),np.array(D, dtype = int)
        return 2/v.shape[0]**2 * np.sum(D**2/(v + 1))
    def obj(M):
        def _obj(p):
            p_ = np.array(p, dtype = int)
            # change the order
            Mp_ = M[p_,:]
            Mp = Mp_[:,p_]
            # objective
            return Um(Mp)
        return _obj
    
    # initialize
    k = D.shape[0]
    best = (None, None)
    objective = obj(D)
    
    # all permutations
    from sympy.utilities.iterables import multiset_permutations
    for p in multiset_permutations([i for i in range(k)]):
        
        # compute objective
        o = objective(p)
        
        # take lesser
        if best[0] is None or best[0] > o:
            best = (o,p)
    return p

def _czekanowski_tsp_seriate(D, initial_order = None):
    k = D.shape[0]
    
    manager = pywrapcp.RoutingIndexManager(k + 1, 1, k)
    routing = pywrapcp.RoutingModel(manager)
    
    # distance callback
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        
        if from_node == k or to_node == k:
            return 0
        
        d = (from_node - to_node)**2 / (1 + D[from_node, to_node]) * 10**6
        return int(d)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = _strategy['first_solution'] 
    search_parameters.local_search_metaheuristic = _strategy['metaheuristic'] 
    search_parameters.time_limit.seconds = _strategy['time_limit']
    #search_parameters.lns_time_limit.seconds = _strategy['local_time_limit']
    search_parameters.solution_limit = _strategy['solution_limit']
    search_parameters.log_search = _strategy['logs']
    
    if initial_order is not None:
        initial_solution = routing.ReadAssignmentFromRoutes([tuple(initial_order)],True)
        solution = routing.SolveFromAssignmentWithParameters(initial_solution,
                                                             search_parameters)
    else:
        solution = routing.SolveWithParameters(search_parameters)
        
    if not solution:
        raise RuntimeError("TSP not solved!")
    
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    
    #print("Objective:", solution.ObjectiveValue())
    return routes[0][1:-1]

def plot(data, func = None, cols = None, initial_order = None, scales = {'exp': 1, 'lin': 1}, **kw):
    assert(data is not None)
    
    # size
    N = data.shape[0]
    
    # default parameters
    if cols is None:
        cols = np.linspace(1, N, num = N).astype(int)

    # distance
    D = cdist(data, data, func) if func is not None else data
    
    # project distance
    def project(M):
        # scale
        M = (M - np.min(M)) / (np.max(M) - np.min(M))
        # exponential projection
        Ms = scales['lin'] * np.exp(-M/2/scales['exp']**2)
        # threshold the little
        Ms[Ms < scales['thres']*scales['lin']] = 0
        return Ms
    D = project(D)
    
    # seriate using initial order
    if initial_order is not None:
        Di = D[:,initial_order]
        Di = Di[initial_order,:]
        colsi = cols[initial_order]
    else:
        Di,colsi = D, cols
    
    # seriate
    D_order = _czekanowski_tsp_seriate(Di, initial_order)
    D_ = Di[:,D_order]
    D_ = D_[D_order,:]
    
    #print("new:", cols[D_order])
    
    # construct data frame
    def construct_df(M, cols):
        P = {'x': [], 'y': [], 'Distance': []}
        for i,x in enumerate(cols):
            for j,y in enumerate(cols):
                P['x'].append(x)
                P['y'].append(y)
                P['Distance'].append(M[i,j])
        P = pd.DataFrame(P)
        return P
    
    P_ = construct_df(D_, colsi[D_order])
    Pi = construct_df(Di, colsi)
    P = construct_df(D, cols)
    
    # plot
    plt.scatter(P.x, P.y, s = (P.Distance), alpha = .3, c = 'r')
    plt.xticks(rotation=90)
    plt.show()
    plt.scatter(Pi.x, Pi.y, s = (Pi.Distance), alpha = .3, c = 'blue')
    plt.xticks(rotation=90)
    plt.show()
    plt.scatter(P_.x, P_.y, s = (P_.Distance), alpha = .3, c = 'g', **kw)
    plt.xticks(rotation=90)
    #plt.show()
    
    return P_
 
    
# TSP strategy
_strategy = {
    'first_solution': routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE, #ALL_UNPERFORMED
    'metaheuristic': routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH, #GUIDED_LOCAL_SEARCH
    'time_limit': 30,
    'local_time_limit': 10,
    'solution_limit': 100000,
    'logs': False
}
def set_strategy(strategy):
    global _strategy
    _strategy = {**_strategy, **strategy}
    return _strategy

if __name__ == "__main__":
    # get data
    regions = src.regions()
    regions = {k:v for k,v in regions.items() if k[:2].upper() in {'PL','CZ'}}
    centroids = [list(map(float, r['centroid'].split(','))) for r in regions.values()]
    cities = np.array([v['name'] for v in regions.values()])

    # random order
    test_order = [i for i in range(len(cities))]
    test_order = np.random.permutation(test_order)
    print("original:", cities[test_order])

    # initial order
    codes = np.array([v['NUTS3'] for v in regions.values()])
    init = np.argsort(codes[test_order])
    print("initial", (cities[test_order])[init])
    
    # run
    plot(np.array(centroids)[test_order], great_circle, cities[test_order],
         init, {'exp': 9, 'lin': 400, 'thres': .1})





