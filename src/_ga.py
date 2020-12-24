# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 08:10:29 2020

@author: martin
"""

import numpy as np
from numpy.random import choice

import ovfw

def population_score(pop, f, axis = 0):
    """Calculate the score for population."""
    return np.array([f(pop[i]) for i in range(pop.shape[axis])])

def select_parents(pop, fitness, parents):
    idx = np.argsort(-fitness)[:parents]
    return pop[idx,:], fitness[idx]

def gen_children(p1, p2, N = 2):
    
    for i in range(0, N, 2):
        # split
        pivot = np.random.randint(p1.shape[0])
        # split onto parts
        c1a,c1b = p1[:pivot],p1[pivot:]
        c2a,c2b = p2[:pivot],p2[pivot:]
    
        # create occurrence mask
        c1a_mask = np.zeros(p1.shape[0], dtype=bool)
        c2a_mask = np.zeros(p1.shape[0], dtype=bool)
        c1a_mask[c1a],c2a_mask[c2a] = True,True
        # remove the duplicates
        c2b, c2a_ = c2b[~c1a_mask[c2b]],c2a[~c1a_mask[c2a]]
        c1b, c1a_ = c1b[~c2a_mask[c1b]],c1a[~c2a_mask[c1a]]
    
        # append the parts
        c1 = np.concatenate([c1a, c2b, c2a_])
        c2 = np.concatenate([c2a, c1b, c1a_])
        yield c1,c2
        

def crossover(pop, f):
    # create container
    children,d = ovfw.Container(pop.shape[0]),pop.shape[1]
    
    # random pairs
    idx = list(range(pop.shape[0]))
    parent1 = np.random.choice(idx, size = pop.shape[0]//2, replace = False)
    parent2 = np.array(list(set(idx) - set(parent1)))
    np.random.shuffle(parent2)
    
    # odd parents number
    if parent1.shape[0] != parent2.shape[0]:
        raise Exception("Odd number of parents!")
        
    # generate children
    for p1,p2 in zip(parent1, parent2):
        
        for c1,c2 in gen_children(pop[p1,:], pop[p2,:], 10):
            children.add(c1, f(c1))
            children.add(c2, f(c2))
    
        
    return children.to_numpy(score = True)

def mutate(pop, f, mutprob, maxiter = None):
    assert(mutprob >= 0 and mutprob <= 1)
    
    if maxiter is None:
        maxiter = pop.shape[0] / mutprob 
    
    mutants,mutants_score = [],[]
    P,d = pop.shape
    
    # iterate
    for i in range(P):
        
        # is he mutated?
        if np.random.uniform() > mutprob:
            continue
        
        # take the individual
        x = pop[i,:]
        
        # perform 1 - 3 random swaps
        for _ in range(choice(range(3))):
            a,b = choice(range(d), size = 2, replace = False)
            x[a],x[b] = x[b],x[a]
        
        mutants.append(x)
        mutants_score.append(f(x))
    
    return mutants, mutants_score
    
    #mutants = ovfw.Container(N)
    #return mutants.to_numpy(score = True)
            
def war(N, p, c, m, take_best = .5):
    # parse inputs
    (p,ps),(c,cs),(m,ms) = p,c,m
    # concatenate individuals and scores
    x = np.concatenate([p,c,m])
    score = np.concatenate([ps, cs, ms])
    
    # choose the N% best
    n_best = int(N * take_best)
    idx_best = np.argsort(-score)[:n_best]
    
    # choose random out of rest
    n_rest = N - n_best
    idx_rest = choice(list(set(range(N)) - set(idx_best)),
                    size = n_rest, replace = False)
    
    # merge to get survivors
    survivors = np.concatenate([x[idx_best,:],x[idx_rest,:]])
    survivors_score = np.concatenate([score[idx_best],score[idx_rest]])
    return survivors[np.argsort(-survivors_score),:]
    
    
