# -*- coding: utf-8 -*-
"""
Module to generate plots.

@author: Martin Benes
"""

import sys
sys.path.append("src")

from datetime import datetime
from dtw import *
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,10)

import location
import _src
import _covid

def centroid_distance_heatmap(h = 100, name = None, countries = None):
    
    # get selection data only
    if countries is None:
        regions = None
    else:
        regions = {k:v for k,v in _src.regions().items() if k[:2].lower() in countries} 

    # get centroid distances
    M,M_nuts = location.centroid_distance_matrix(h = h, regions = regions)
    # get region names (for legend)
    M_names = _src.region_names(M_nuts)

    # plot
    sns.heatmap(M, xticklabels=M_names, yticklabels=M_names)
    if name is None: plt.show()
    
    # save plot
    else: plt.savefig(name)

def centroid_distance_heatmap_czpl(*args, **kw):
    # plot with explicit data
    centroid_distance_heatmap(*args, countries = {'cz','pl'}, **kw)
    
def centroid_distance_heatmap_se(*args, **kw):
    # plot with explicit data
    centroid_distance_heatmap(*args, countries = {'se'}, **kw)

def adjacency_matrix(name = None, countries = None):
    
    # get selection data only
    if countries is None:
        neighbors = None
    else:
        neighbors = {k:v for k,v in _src.neighbors().items() if k[:2].lower() in countries}
    
    # adjacencies
    A,A_nuts = location.adjacency_matrix(neighbors = neighbors)
    # get region names (for legend)
    A_names = _src.region_names(A_nuts)
    
    # discrete colormap
    cmap = sns.color_palette("gray_r", 2)  
    
    # plot
    ax = sns.heatmap(A, xticklabels=A_names, yticklabels=A_names, cmap=cmap)
    # modify colorbar:
    colorbar = ax.collections[0].colorbar 
    r = colorbar.vmax - colorbar.vmin 
    colorbar.set_ticks([0.,1.])
    colorbar.set_ticklabels(["Not adjacent","Adjacent"])                                          
    
    if name is None: plt.show()
    
    # save plot
    else: plt.savefig(name)

def adjacency_matrix_czpl():
    # plot with explicit data
    adjacency_matrix(countries = {'cz','pl'})

def adjacency_heatmap(name = None, countries = None):
    
    # get selection data only
    if countries is None:
        neighbors = None
    else:
        neighbors = {k:v for k,v in _src.neighbors().items() if k[:2].lower() in countries}

    # get adjacencies
    M,M_nuts = location.adjacency_similarity_matrix(neighbors = neighbors)
    # get region names (for legend)
    M_names = _src.region_names(M_nuts)
    
    # plot
    sns.heatmap(M, xticklabels=M_names, yticklabels=M_names)
    if name is None: plt.show()
    
    # save plot
    else: plt.savefig(name)

def adjacency_heatmap_czpl():
    # plot with explicit data
    adjacency_heatmap(countries = {'cz','pl'})
    
def adjacency_heatmap_se():
    # plot with explicit data
    adjacency_heatmap(countries = {'se'})

def location_score_heatmap(h = 100, name = None, countries = None):
    
    # get selection data only
    if countries is None:
        regions = None
    else:
        regions = {k:v for k,v in _src.regions().items() if k[:2].lower() in countries} 
    
    # get score
    M,M_nuts = location.location_score_matrix(h = h, regions = regions) # bug
    # get region names
    M_names = _src.region_names(M_nuts)
    
    # plot
    sns.heatmap(M, xticklabels=M_names, yticklabels=M_names)
    if name is None: plt.show()
    
    # save plot
    else: plt.savefig(name)

def location_score_heatmap_czpl(h = 100):

    # plot with explicit data
    location_score_heatmap(h = h, countries = {'cz','pl'})
    
def location_score_heatmap_se(h = 100):
    
    # plot with explicit data
    location_score_heatmap(h = h, countries = {'se'})

def area_population_scatter(name = None, regions_df = None):
    
    # default data if not given
    regions_df = regions_df if regions_df is not None else _src.regions_df()
    
    # plot
    ax = sns.jointplot(x = "Area", y = "Population", hue="Country", data=regions_df)
    # axis limits
    ax.ax_marg_x.set_xlim(-1.5*10**4, 1.25*10**5)
    ax.ax_marg_y.set_ylim(-5*10**5, 6*10**6)
    if name is None: plt.show()
    
    # save
    else: plt.savefig(name)

def popdensity_boxplot(name = None, regions_df = None):
    
    # default data if not given
    regions_df = regions_df if regions_df is not None else _src.regions_df()
    
    # plot
    plt.rcParams.update({'font.size': 20})
    #plt.yscale("log")
    sns.violinplot(x="Country", y="Density", data=regions_df)
    #sns.boxplot(x="Country", y="Density", data=regions_df, color = "1")
    sns.stripplot(x="Country", y="Density", color='black', size=6, alpha=0.8, data=regions_df)
    if name is None: plt.show()
    
    # save
    else: plt.savefig("density.png")
    

def popdensity_boxplot_noPRG(name = None, regions_df = None):
    
    # default data if not given
    regions_df = regions_df if regions_df is not None else _src.regions_df()
    
    # crop prague off
    regions_df = regions_df[regions_df.Code != "CZ010"]
    
    # plot
    popdensity_boxplot(name = name, regions_df = regions_df)
 
_dtwcache = {}
_seed = 54321
def _dtw_data(*args, **kw):
    global _dtwcache, _seed
    key = str(args) + str(kw)
    try: return _dtwcache[key]
    except:
        np.random.seed = _seed
        _dtwcache[key] = _covid.czekanowski_dtw(*args, **kw)
    return _dtwcache[key]
def flush_cache(seed = 54321):
    global _dtwcache, _seed
    _dtwcache = {}
    _seed = seed
    
def deaths_dtw_czekanowski(*args, **kw):
    
    # data
    P = _dtw_data(*args, **kw)
    
    # plot
    plt.rcParams.update({'font.size': 12})
    plt.scatter(P.x, P.y, s = (P.Distance), alpha = .9, c = 'black')
    plt.xticks(rotation=90)
    plt.show()

def deaths_country_series():
    
    # read data
    data = _covid.deaths_df()
    
    # data by country
    data['country'] = data.region\
        .apply(lambda i: i[:2])
    data = data\
        .groupby(['date','country'])\
        .aggregate({'deaths': 'sum'})\
        .reset_index()
    
    # plot
    plt.rcParams.update({'font.size': 16})
    sns.lineplot(data = data, x = "date", y = "deaths", hue = "country")
    plt.plot(2*[datetime(2020,7,31)], [0, data.deaths.max()], c = "black")
    plt.plot(2*[data.date.min()], [0, data.deaths.max()], c = "black")
    plt.show()
    
def deaths_dtw_czekanowski_global():
    deaths_dtw_czekanowski(h = .008, coef = 300, random_starts = 1)

def deaths_dtw_czekanowski_firstWave():
    
    # read data
    data = _covid.deaths_df()
    # filter (only first wave)
    data = data[data.date < datetime(2020, 8, 1)]
    
    # czekanowski of the first wave
    deaths_dtw_czekanowski(data = data, h = .008, coef = 250, random_starts = 1)

def deaths_dtw_czekanowski_sweden():
    deaths_dtw_czekanowski(h = .035, coef = 300, random_starts = 1)

def deaths_dtw_czekanowski_sweden_firstWave():
    
    # read data
    data = _covid.deaths_df()
    # filter (only first wave)
    data = data[data.date < datetime(2020, 8, 1)]
    
    # czekanowski of the first wave
    deaths_dtw_czekanowski(data = data, h = .035, coef = 300, random_starts = 1)
    

def deaths_dtw_heatmap(*args, **kw):
    
    # data
    P = _dtw_data(*args, **kw)
    
    # df to np
    d = int(np.sqrt(P.Distance.shape[0]))
    PDist_np = P.Distance.to_numpy().reshape((d,d))
    lab = P.y[:d].to_list()
    PDist_np = np.flip(PDist_np, axis = 0)
    
    # plot
    sns.heatmap(PDist_np, xticklabels=lab, yticklabels=list(reversed(lab)))
    plt.show()

def deaths_smooth(region):
    
    # fetch data
    x,y,fx = _covid.deaths_smooth(region = region)

    # plot
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    ax.plot(x, fx, c = 'r')
    plt.show()
    
def dtw_plot(x, y, *args, **kw):
    
    # data
    deaths = _covid.deaths_df()
    
    # smooth
    plt.rcParams.update({'font.size': 16})
    x1,x2,fx = _covid.deaths_smooth(x, deaths)
    y1,y2,fy = _covid.deaths_smooth(y, deaths)

    # map names onto code
    dtw(fx, fy, keep_internals=True, 
        step_pattern=rabinerJuangStepPattern(6, "c"))\
        .plot(type="twoway")
    plt.show()

def weekday_ratio_heatmap():
    
    # fetch data
    D,lab = _covid.weekday_ratio_distance()
    
    # plot
    plt.rcParams.update({'font.size': 16})
    sns.heatmap(D, xticklabels=lab, yticklabels=lab)
    plt.show()

