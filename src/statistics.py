# -*- coding: utf-8 -*-
"""
Module to perform statistical analysis.

@author: Martin Benes
"""

import sys
sys.path.append('src')

import math
import numpy as np
from scipy.stats import ttest_ind, levene, f_oneway

import _src

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

