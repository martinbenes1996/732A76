# -*- coding: utf-8 -*-
"""
Module to generate regional location statistics of regions.

@author: Martin Benes
"""

import sys
sys.path.append("src")

import csv
import numpy as np
from scipy.spatial.distance import cdist, correlation
from seriate import seriate
from sklearn.preprocessing import minmax_scale

import _src
import tools

def adjacency_matrix(neighbors = None):
    
    # default data
    neighbors = neighbors if neighbors is not None else _src.neighbors()
    
    # compute adjacency
    regIdx = {n: i for i,n in enumerate(neighbors)}
    adjacent = np.zeros((len(neighbors),len(neighbors)))
    for k1,n in neighbors.items():
        for k2 in n:
            adjacent[regIdx[k1],regIdx[k2]] = 1
    
    # seriate
    D_order = seriate(adjacent)
    adjacent_ = adjacent[:,D_order]
    adjacent_ = adjacent_[D_order,:]
    
    # region labels
    region_nuts = np.array(list(neighbors.keys()))[D_order]
    
    # return
    return adjacent_,region_nuts

def adjacency_similarity_matrix(neighbors = None):
    
    # default data
    neighbors = neighbors if neighbors is not None else _src.neighbors()
    
    # compute adjacency
    regIdx = {n: i for i,n in enumerate(neighbors)}
    adjacent = np.zeros((len(neighbors),len(neighbors)))
    for k1,n in neighbors.items():
        for k2 in n:
            adjacent[regIdx[k1],regIdx[k2]] = 1
    
    # distance matrix
    D_adjacent = cdist(adjacent, adjacent, metric = correlation)
    # seriate
    D_order = seriate(D_adjacent)
    D_adjacent_ = D_adjacent[:,D_order]
    D_adjacent_ = D_adjacent_[D_order,:]
    
    # region labels
    region_nuts = np.array(list(neighbors.keys()))[D_order]
    
    # return
    return D_adjacent_,region_nuts

def centroid_distance_matrix(h = 100, regions = None):
    
    # default data
    regions = regions if regions is not None else _src.regions()
    
    # parse centroids
    centroids = [list(map(float, r['centroid'].split(','))) for r in regions.values()]
    
    # distance matrix
    D_centroid = cdist(centroids, centroids, metric = tools.great_circle)
    # Gaussian kernel (distance -> similarity matrix)
    K_centroid = tools.rbf(D_centroid, h = h)
    # scale inverted (similarity -> distance matrix)
    K_centroid = minmax_scale(-K_centroid)
    # seriate
    K_order = seriate(K_centroid)
    K_centroid_ = K_centroid[:,K_order]
    K_centroid_ = K_centroid_[K_order,:]
    
    # region labels
    region_nuts = np.array(list(regions.keys()))[K_order]
    
    # return
    return K_centroid_,region_nuts
    
def location_score_matrix(h = 100, regions = None):
    
    # default data
    regions = regions if regions is not None else _src.regions()
    
    # get neighbors
    neighbors = {k:v for k,v in _src.neighbors().items() if k in regions}
    
    # construct matrices
    M_ctr,ctr_nuts = centroid_distance_matrix(h = h, regions = regions)
    M_adj,adj_nuts = adjacency_similarity_matrix(neighbors = neighbors)
    
    # reorder to have same order
    adj_reorder = [i for c in ctr_nuts for i,a in enumerate(adj_nuts) if a == c]
    M_adj_ = M_adj[:,adj_reorder]
    M_adj_ = M_adj_[adj_reorder,:]
    adj_nuts_ = adj_nuts[adj_reorder]
    
    # construct score
    neighbors_weighted = np.sqrt(M_adj_ * M_ctr)
    
    # seriate
    neighbors_order = seriate(neighbors_weighted)
    neighbors_weighted_ = neighbors_weighted[:,neighbors_order]
    neighbors_weighted_ = neighbors_weighted_[neighbors_order,:]
    
    # region labels
    neighbors_nuts = np.array(adj_nuts_)[neighbors_order]
    
    # return
    return neighbors_weighted_,neighbors_nuts

