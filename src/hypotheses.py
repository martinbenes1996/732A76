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
from scipy.stats import ttest_ind, levene, t, kstest

import _covid
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
    descriptive_cols = {'Region','Code','Country'}
    
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

class Distr:
    def __init__(self, pvalue):
        self.pvalue = pvalue
        
def is_t_distributed(X, K = 500):
    # get t parameters
    nu,mu,sigma = t.fit(X, loc=X.mean(), scale=X.std())
    # Kolmogorov-Smirnoff of original sample
    stat0,_ = kstest(X.to_numpy(), t.cdf, args = (nu,))
    
    # distribution
    d = []
    for k in range(K):
        # generate
        tsample = t.rvs(nu, loc=mu, scale=sigma, size=X.shape[0])
        # KS
        stat,_ = kstest(tsample, t.cdf, args = (nu,))
        d.append(stat)
    d = np.array(d)
    
    # compute pvalue
    pvalue = (np.sum(d > stat0) + 1) / (d.shape[0] + 1)
    return Distr(pvalue)

def regions_t_distributed(K = 500, pi = True, alpha = .05):
    """Tests whether regions are t-distributed in their
    populations, areas and densities over different countries.
    Uses sample test.
    
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
            
            # t test
            pi_df.at[i,a] = is_t_distributed(data, K = K).pvalue
            
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
    """Tests that regions have same mean in their populations, areas and densities.
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

def weekdays_equal_ratio(pi = True, alpha = .05):
    """Tests that deaths have equal ratio over week days.
    Use two-sampled t_test.
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
    x = _covid.deaths_df()
    # years and weekdays
    x['year'] = x.date.apply(lambda d: d.year)
    x['weekday'] = x.date.apply(lambda d: d.weekday() + 1)
    x['country'] = x.region.apply(lambda d: d[:2])
    
    # over all regions
    x = x[["country","year","week","weekday","deaths"]]\
        .groupby(["country","year","week","weekday"])\
        .sum()\
        .reset_index()
    x.columns = ["country","year","week","weekday", *x.columns[4:]]
    # over everything
    total = x\
        .groupby(['country','week'])\
        .aggregate({'deaths': 'sum'})\
        .reset_index()\
        .rename(mapper = {'deaths': 'total'}, axis = 1)
    x = x.merge(total, on = ["country","week"])
    x['deaths'] = (x.deaths / x.total)\
        .fillna(0)
    
    # create dataframe
    weekdays = ['Mo','Tu','We','Th','Fr','Sa','Su']
    pi_dict = {k: [None for _ in range(len(x.country.unique()))] for k in weekdays}
    pi_dict = {
        'Country': [c for c in x.country.unique()],
        **pi_dict
    }
    pi_df = pd.DataFrame(pi_dict)
    
    # per country
    for gname, g in x.groupby('country'):
        # prepare t-test
        gmean = g\
            .groupby('weekday')\
            .aggregate({'deaths': 'sum'})
        gmean['deaths'] /= gmean.deaths.sum()
        gvar = g\
            .groupby('weekday')\
            .aggregate({'deaths': 'var'})
        gN = g[['year','week']]\
            .drop_duplicates()\
            .shape[0]
        # t statistic
        tt = (gmean - 1/7) / np.sqrt(gvar / gN)
        pval = (t.sf(np.abs(tt), gN-1)*2)
        
        # result
        for i,a in enumerate(weekdays):
            # value
            val = pval if pi else pval > alpha
            # set
            row = (pi_df.Country == gname)
            #print(pi_df[a,np.argmax(row)])
            pi_df[a][np.argmax(row)] = val[i,0]
    
    # return
    return pi_df
            
