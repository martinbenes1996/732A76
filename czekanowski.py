# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 08:33:52 2020

@author: martin
"""

import numpy as np
from numpy.random import permutation
import pandas as pd

import logging
LOG = logging.getLogger(__name__)

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9,7)
plt.rcParams['font.size'] = 7

import ga
import ovfw

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



def _czekanowski_ga_seriate(D, popsize = 30, maxiter = 1000, mutprob = .1,
                            eps = .001, MA_size = 150, random_starts = 1):
    # input conditions
    assert(len(D.shape) == 2 and D.shape[0] == D.shape[1])
    assert(popsize > 1)
    assert(maxiter > 0)
    assert(eps > 0)
    assert(MA_size > 0)
    assert(random_starts > 0)
    
    # initialize parameteres
    d = D.shape[0]
    N = maxiter
    Nparents = int(popsize * .9)
    if Nparents % 2 != 0:
        Nparents -= 1
        
    # objective function
    o = obj(D)
    
    LOG.info("== Permutation-estimating GA ==")
    LOG.info("Input shape: (%d,%d)" % D.shape)
    LOG.info("Pop shape: (%d,%d)" % (popsize,d))
    LOG.info("Generation: [%d iterations]" % maxiter)
    LOG.info("| Size:    %d" % popsize)
    LOG.info("| Parents: %d [%5.3f%% of population]" % (Nparents, Nparents/popsize))
    
    optimal,optimal_score = None,float("-inf")
    for r in range(random_starts):
        # initialize population
        pop = np.array([permutation([i for i in range(d)]) for _ in range(popsize)])
    
        # run
        MA_window = ovfw.Container(MA_size)
        for it,generation in enumerate(range(N)):
        
            # fitness of each chromosome
            fitness = ga.population_score(pop, o)
            
            if (it == 0 or (it + 1) % 100 == 0):
                LOG.info("Iteration %d: score %5.3f" % (it + 1, (-np.sort(-fitness))[0]))
            
            # crossover
            parents,pscore  = ga.select_parents(pop, fitness, Nparents)
            children,cscore = ga.crossover(parents, o)
        
            # create mutants
            mutants,mscore = ga.mutate(children, o, mutprob)
        
            # war
            pop = ga.war(popsize, (parents,pscore),(children,cscore),(mutants,mscore))
        
            # early stopping
            best,best_score = pop[0,:],o(pop[0,:])
            MA_window.add(it, best_score)
            if it > MA_size:
                if np.mean(np.abs(MA_window.score() - best_score)) < eps:
                    LOG.info("Early stopping!")
                    break
        
        if best_score > optimal_score:
            optimal = best
            optimal_score = best_score
        
        LOG.info("Random start %d: best score %5.3f" % (r + 1, optimal_score))
        
    return optimal

def distance_rbf(data, func = None,
                 h:float = 1, coef:float = 1, cutoff:float = 0.1):
    """Compute the distance matrix, centers and transform with RBF (Gaussian kernel).
    
    Args:
        data (pd.Dataframe): Input.
        func (callable): Distance metric, passed to cdist.
        h (float, optional): Kernel width, 1 by default. If None, kernel is omitted. 
    """
    assert(data is not None)
    assert(coef is not None)
    assert(cutoff is not None)
    
    # distance
    D = cdist(data, data, func) if func is not None else data
    
    # standardization
    #D = (D - np.mean(D)) / np.std(D)
    
    # kernel transformation
    if h is not None:
        D = np.exp(-(D/h)**2/2)
    D /= D.max()
    D *= coef
    
    # cutoff
    
    D[D <= np.quantile(D, cutoff)] = 0
    
    return D

def plot(data, cols = None, diagonal = False, **kw):
    assert(data is not None)
    
    # size
    N = data.shape[0]
    
    # default parameters
    if cols is None:
        cols = np.linspace(1, N, num = N).astype(int)
    
    # construct data frame
    def construct_df(M, cols, diagonal = False):
        P = {'x': [], 'y': [], 'Distance': []}
        for i,x in enumerate(cols):
            for j,y in enumerate(cols):
                if not diagonal and x == y: continue
                P['x'].append(x)
                P['y'].append(y)
                P['Distance'].append(M[i,j])
        P = pd.DataFrame(P)
        return P
    
    # plot original data
    P = construct_df(data, cols, diagonal = diagonal)
    plt.scatter(P.x, P.y, s = (P.Distance), alpha = .9, c = 'red')
    plt.xticks(rotation=90)
    plt.show()
    
    # estimate permutation using GA
    D_order = _czekanowski_ga_seriate(data, **kw)
    # permute
    data = data[:,D_order]
    data = data[D_order,:]
    
    # plot new permutation
    P = construct_df(data, cols[D_order], diagonal = diagonal)
    plt.scatter(P.x, P.y, s = (P.Distance), alpha = .9, c = 'g')
    plt.xticks(rotation=90)
    plt.show()
    
    return P
 


