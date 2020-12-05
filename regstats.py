# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 22:39:54 2020

@author: martin
"""

import math
import numpy as np
from scipy.stats import ttest_ind
import src


def outliers(feature, country = None):
    
    # fetch region dataframe
    x = src.regions_df()
    
    # subset country
    if country:
        x = x[x.Country.str.lower() == country.lower()]
    
    # IQR
    q1,q3 = np.percentile(sorted(x[feature]),[25,75])
    iqr = q3 - q1
    low,high = (q1 - (1.5*iqr), q3 + (1.5*iqr))
    outs = x[(x[feature] < low) | (x[feature] > high)]
    
    # return outliers
    return outs

#for country in ['cz','pl','se']:
#    for attr in ['Area','Population','Density']:
#        print("===== %s [%s] =====" % (country,attr))
#        print(outliers(attr, country))
#        print("")

def Welchs_test(pi = True, alpha = 0.95):
    """Performs Welchs tests of attributes from regions of given countries.
    
    Args:
        countries (list): List of country codes as strings.
        attributes (list): List of attributes as strings.
    """
    
    # fetch region dataframe
    regions_df = src.regions_df()
    
    # default values
    attributes = ['Population','Area','Density']
    countries = regions_df.Country.unique()
    
    # create dataframe
    pi_dict = {k: [None for _ in range(len(countries)**2)] for k in attributes}
    pi_dict = {
        'Country1': [c for c in countries for _ in range(len(countries))],
        'Country2': [c for _ in range(len(countries)) for c in countries],
        **pi_dict
    }
    
    # pi values dataframes
    pi_df = pd.DataFrame(pi_dict)
    pi_df = pi_df[pi_df.Country1 != pi_df.Country2]\
        .reset_index(drop = True)
    
    for i,r in pi_df.iterrows():
        for a in attributes:
            # get data
            data1 = regions_df[regions_df.Country == r.Country1][a]
            data2 = regions_df[regions_df.Country == r.Country2][a]
            
            # perform the test
            pop_pi = ttest_ind(data1, data2, equal_var = False)
            pi_df.at[i,a] = pop_pi.pvalue
    
    if pi:
        return pi_df
    else:
        pi_df = pd.concat([
            pi_df[['Country1','Country2']],
            pi_df[attributes] < (1 - alpha)/2
        ], axis = 1, ignore_index=True)
        pi_df.columns = ['Country1','Country2', *attributes]
        return pi_df
    
Welchs_test()
Welchs_test(pi = False)


# population
# SE - PL
pop_pi = ttest_ind(regions_se.Population, regions_pl.Population, equal_var = False)
pop_pi.pvalue
print("For Population SE-PL, H0 is", pop_pi.pvalue > 0.025)
# SE - CZ
pop_pi = ttest_ind(regions_se.Population, regions_cz.Population, equal_var = False)
pop_pi.pvalue
print("For Population SE-CZ, H0 is", pop_pi.pvalue > 0.025)
# CZ - PL
pop_pi = ttest_ind(regions_cz.Population, regions_pl.Population, equal_var = False)
pop_pi.pvalue
print("For Population CZ-PL, H0 is", pop_pi.pvalue > 0.025)
# area
# SE - PL
area_pi = ttest_ind(regions_se.Area, regions_pl.Area, equal_var = False)
area_pi.pvalue
print("For Area SE-PL, H0 is", area_pi.pvalue > 0.025)
# SE - CZ
area_pi = ttest_ind(regions_se.Area, regions_cz.Area, equal_var = False)
area_pi.pvalue
print("For Area SE-CZ, H0 is", area_pi.pvalue > 0.025)
# CZ - PL
area_pi = ttest_ind(regions_cz.Area, regions_pl.Area, equal_var = False)
area_pi.pvalue
print("For Area CZ-PL, H0 is", area_pi.pvalue > 0.025)
# density
# SE - PL
density_pi = ttest_ind(regions_se.Density, regions_pl.Density, equal_var = False)
density_pi.pvalue
print("For Density SE-PL, H0 is", density_pi.pvalue > 0.025)
# SE - CZ
density_pi = ttest_ind(regions_se.Density, regions_cz.Density, equal_var = False)
density_pi.pvalue
print("For Density SE-CZ, H0 is", density_pi.pvalue > 0.025)
# CZ - PL
density_pi = ttest_ind(regions_cz.Density, regions_pl.Density, equal_var = False)
density_pi.pvalue
print("For Density CZ-PL, H0 is", density_pi.pvalue > 0.025)
