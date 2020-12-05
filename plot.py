
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9,7)

import location
import src


def centroid_distance_heatmap(name = None, countries = None):
    
    # get selection data only
    if countries is None:
        regions = None
    else:
        regions = {k:v for k,v in src.regions().items() if k[:2].lower() in countries} 

    # get centroid distances
    M,M_nuts = location.centroid_distance_matrix(regions = regions)
    # get region names (for legend)
    M_names = src.region_names(M_nuts)

    # plot
    sns.heatmap(M, xticklabels=M_names, yticklabels=M_names)
    if name is None: plt.show()
    
    # save plot
    else: plt.savefig(name)

def centroid_distance_heatmap_czpl():
    # plot with explicit data
    centroid_distance_heatmap(countries = {'cz','pl'})
    
def centroid_distance_heatmap_se():
    # plot with explicit data
    centroid_distance_heatmap(countries = {'se'})

def adjacency_heatmap(name = None, countries = None):
    
    # get selection data only
    if countries is None:
        neighbors = None
    else:
        neighbors = {k:v for k,v in src.neighbors().items() if k[:2].lower() in countries}

    # get adjacencies
    M,M_nuts = location.adjacency_matrix(neighbors = neighbors)
    # get region names (for legend)
    M_names = src.region_names(M_nuts)
    
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

def location_score_heatmap(name = None, countries = None):
    
    # get selection data only
    if countries is None:
        regions = None
    else:
        regions = {k:v for k,v in src.regions().items() if k[:2].lower() in countries} 
    
    # get score
    M,M_nuts = location.location_score_matrix(regions = regions)
    # get region names
    M_names = src.region_names(M_nuts)
    
    # plot
    sns.heatmap(M, xticklabels=M_names, yticklabels=M_names)
    if name is None: plt.show()
    
    # save plot
    else: plt.savefig(name)

def location_score_heatmap_czpl():

    # plot with explicit data
    location_score_heatmap(countries = {'cz','pl'})
    
def location_score_heatmap_se():
    
    # plot with explicit data
    location_score_heatmap(countries = {'se'})

def area_population_scatter(name = None, regions_df = None):
    
    # default data if not given
    regions_df = regions_df if regions_df is not None else src.regions_df()
    
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
    regions_df = regions_df if regions_df is not None else src.regions_df()
    
    # plot
    plt.yscale("log")
    sns.boxplot(x="Country", y="Density", data=regions_df, color = "1")
    sns.stripplot(x="Country", y="Density", color='black', size=10, alpha=0.5, data=regions_df)
    if name is None: plt.show()
    
    # save
    else: plt.savefig("density.png")
    

def popdensity_boxplot_noPRG(name = None, regions_df = None):
    
    # default data if not given
    regions_df = regions_df if regions_df is not None else src.regions_df()
    
    # crop prague off
    regions_df = regions_df[regions_df.Code != "CZ010"]
    
    # plot
    popdensity_boxplot(name = name, regions_df = regions_df)

#popdensity_boxplot()
#popdensity_boxplot_noPRG()