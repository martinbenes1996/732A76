# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 08:33:52 2020

@author: martin
"""

import sys
sys.path.append('src')

import numpy as np
from numpy.random import permutation
import pandas as pd

import logging
LOG = logging.getLogger(__name__)

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9,7)
plt.rcParams['font.size'] = 7

import _ga
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
                            eps = .001, MA_size = 150, random_starts = 1, **kw):
    # input conditions
    assert(len(D.shape) == 2 and D.shape[0] == D.shape[1])
    assert(popsize > 1)
    assert(maxiter > 0)
    assert(eps > 0)
    assert(MA_size > 0)
    assert(random_starts > 0)
    
    # initialize parameteres
    d = D.shape[0]
    Nparents = int(popsize*.75)
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
        for it,generation in enumerate(range(maxiter)):
        
            # fitness of each chromosome
            fitness = _ga.population_score(pop, o)
            
            if (it == 0 or (it + 1) % 100 == 0):
                LOG.info("Iteration %d: score %5.3f" % (it + 1, (-np.sort(-fitness))[0]))
            
            # crossover
            parents,pscore  = _ga.select_parents(pop, fitness, Nparents)
            children,cscore = _ga.crossover(parents, o)
        
            # create mutants
            mutants,mscore = _ga.mutate(children, o, mutprob)
        
            # war
            pop = _ga.war(popsize, (parents,pscore),(children,cscore),(mutants,mscore))
        
            # early stopping
            best,best_score = pop[0,:],o(pop[0,:])
            if it > MA_size:
                if np.mean(np.abs(MA_window.score() - best_score)) < eps:
                    LOG.info("Early stopping!")
                    break
            
            # append to MA window
            MA_window.add(it, best_score)
            
        if best_score > optimal_score:
            optimal = best
            optimal_score = best_score
        
        LOG.info("Random start %d: best score %5.3f" % (r + 1, optimal_score))
        
    return optimal


import numpy as np




def _czekanowski_olo_seriate(D, **kw):
    # input conditions
    assert(len(D.shape) == 2 and D.shape[0] == D.shape[1])

    # import R
    from rpy2.robjects.packages import importr
    import rpy2.robjects as ro
    import rpy2.robjects.numpy2ri
    utils = importr('utils')
    rpy2.robjects.numpy2ri.activate()
    # install seriation
    utils.chooseCRANmirror(ind=1)
    utils.install_packages("seriation")
    seriation = importr('seriation')
    
    # move data to R 
    #D = np.random.random(size = (50,50))
    nr,nc = D.shape
    M = ro.r.matrix(D, nrow=nr, ncol=nc)
    Dist = ro.r['as.dist'](M)
    # perform seriation
    x = seriation.seriate(Dist, method = "GW")
    order = np.array(seriation.get_order(x))
    
    # return order
    return order - 1


def distance_rbf(data, func = None,
                 h:float = 1, coef:float = 1, cutoff:float = 0.1, **kw):
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

def plot(data, method = "OLO", cols = None, diagonal = False, **kw):
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
    #P = construct_df(data, cols, diagonal = diagonal)
    #plt.scatter(P.x, P.y, s = (P.Distance), alpha = .9, c = 'red')
    #plt.xticks(rotation=90)
    #plt.show()
    
    if method == "OLO":
        # estimate permutation using OLO
        D_order = _czekanowski_olo_seriate(data, **kw)
        print(D_order)
        # kernel projection
        LOG.info("computing distance kernel")
        kdata = distance_rbf(
            data,
            **kw
        )
        
    elif method == "Um":
        # kernel projection
        LOG.info("computing distance kernel")
        kdata = distance_rbf(
            data,
            **kw
        )
        # estimate permutation using GA
        D_order = _czekanowski_ga_seriate(kdata, **kw)
    
    # permute
    kdata = kdata[:,D_order]
    kdata = kdata[D_order,:]
    
    # plot new permutation
    P = construct_df(kdata, cols[D_order], diagonal = diagonal)
    
    return P
    #plt.scatter(P.x, P.y, s = (P.Distance), alpha = .9, c = 'g')
    #plt.xticks(rotation=90)
    #plt.show()
    
    #return P
 
def heatmap(data, cols = None, diagonal = False, **kw):
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
    
    def plot_heatmap(P):
        # df to np
        d = int(np.sqrt(P.Distance.shape[0]))
        PDist_np = P.Distance.to_numpy().reshape((d,d))
        lab = P.y[:d].to_list()
        PDist_np = np.flip(PDist_np, axis = 0)
        # plot
        sns.heatmap(PDist_np, xticklabels=lab, yticklabels=list(reversed(lab)))
        plt.show()
    # plot original data
    P = construct_df(data, cols, diagonal = diagonal)
    plot_heatmap(P)
    
    # estimate permutation using GA
    D_order = _czekanowski_ga_seriate(data, **kw)
    # permute
    data = data[:,D_order]
    data = data[D_order,:]
    
    # plot new permutation
    P = construct_df(data, cols[D_order], diagonal = diagonal)
    plot_heatmap(P)
    
    return P

