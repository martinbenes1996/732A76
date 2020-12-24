# -*- coding: utf-8 -*-
"""
Module to perform statistical analysis.

@author: Martin Benes
"""

import sys
sys.path.append('src')

import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, levene, f_oneway

import _src
import _tools

def IQR(regions = None, countries = None):
    """IQR method for region dataframe.
    
    Args:
        regions (pd.DataFrame, optional): Optional data to work on.
                                          If not given, all regions used.
        countries ()
    """
    
    # fetch region dataframe
    regions = _src.regions_df() if regions is None else regions
    
    # subset country
    if countries:
        regions = regions[regions.Country.isin(list(countries))]
    
    def _IQR(x):
        """IQR method for a vector."""
        q1,q3 = np.percentile(sorted(x),[25,75])
        iqr = q3 - q1
        low,high = (q1 - (1.5*iqr), q3 + (1.5*iqr))
        return (x < low) | (x > high)
    
    # remove descriptive columns
    outs = {}
    for c in set(regions.columns) - descriptive_cols:
        flags = _IQR(regions[c])
        outs[c] = regions[flags]
    
    # return outliers
    return outs

def regions_normally_distributed(pi = True, alpha = .05):
    """Tests whether regions are distributed normally in their
    populations, areas and densities over different countries.
    Uses Shapiro-Wilk test.
    
    H0: Regions of country are distributed normally in the attribute.
    HA: They are not.
    
    Args:
        pi (bool): If True, returns pi values.
                   If False, returns validity of H0.
                   Defautly True.
        alpha (float): Significance level.
    """
    
    # data
    attributes = ['Population','Area','Density']
    regions_df = _src.regions_df()
    
    # empty single country
    countries = regions_df.Country.unique()
    # create dataframe
    pi_dict = {k: [None for _ in range(len(countries))] for k in attributes}
    pi_dict = {
        'Country': [c for c in countries],
        **pi_dict
    }
    # pi values dataframes
    pi_df = pd.DataFrame(pi_dict)
    
    # perform the test
    for a in attributes:
        for i,r in pi_df.iterrows():    
            # data
            data = regions_df[regions_df.Country == r.Country][a]
            
            # Shapiro-Wilk test
            normal_pi = stats.shapiro(data)
            pi_df.at[i,a] = normal_pi.pvalue

    # return pi value
    if pi: return pi_df
    
    # make decision
    pi_df = pd.concat([
        pi_df[['Country']],
        pi_df[attributes] > alpha
    ], axis = 1, ignore_index = True)
    pi_df.columns = ['Country', *attributes]
    return pi_df

def administrative_divisions_similar(pi = True, alpha = .05):
    """Tests that regions of two countries have same mean
    in their populations, areas and densities.
    Use two-sampled t_test with preceding F-test to test equal variances.
    Only the result of t_test is returned.
    
    H0: mu1 = mu2
    HA: mu1 != mu2
    
    Args:
        pi (bool): If True, returns pi values.
                   If False, returns validity of H0.
                   Defautly True.
        alpha (float): Significance level.
    """

    # data
    attributes = ['Population','Area','Density']
    regions_df = _src.regions_df()
    pi_df = _src.regions_countries_pairs(attributes)
    
    # perform the test
    for a in attributes:
        for i,r in pi_df.iterrows():    
            # data
            data1 = regions_df[regions_df.Country == r.Country1][a]
            data2 = regions_df[regions_df.Country == r.Country2][a]
            
            # F-test
            #ftest, fpi = _tools.f_test(data1, data2)
            ftest,fpi = stats.levene(data1, data2)
            equal_var = fpi > alpha
            #print(np.std(data1), np.std(data2))
            #print(r.Country1, r.Country2, a, fpi, equal_var)
            
            # test
            pop_pi = stats.ttest_ind(data1, data2, equal_var = equal_var)
            # write down pvalue
            pi_df.at[i,a] = pop_pi.pvalue
    
    # return pi value
    if pi: return pi_df
    
    # make decision
    pi_df = pd.concat([
        pi_df[['Country1','Country2']],
        pi_df[attributes] > alpha
    ], axis = 1, ignore_index = True)
    pi_df.columns = ['Country1','Country2', *attributes]
    return pi_df


