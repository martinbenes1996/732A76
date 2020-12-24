# -*- coding: utf-8 -*-
"""
Module containing tools used globally among many modules.

@author: Martin Benes
"""

from math import radians, sin, cos, acos
import numpy as np
import pandas as pd
from scipy.stats import f

import eunuts
#eunuts.init_cache()

def nuts(city, country):
    """Map city onto NUTS code.
    
    Args:
        city (str): City name.
        country (str): Country where the city is.
    Returns:
        (str) NUTS code of the city.
    """
    df = pd.DataFrame(data = {'city': city, 'country': country})
    return eunuts.nuts(df)


def great_circle(coords1, coords2):
    """Great circle distance / Haversine formula.
    
    Args:
        coords1,coords2 (tuple): Latitude and longitude in degrees, floats.
    Returns:
        (float) Distance between the two points in km.
    """
    lon1, lat1, lon2, lat2 = map(radians, (*coords1, *coords2))
    if lon1 == lon2 and lat1 == lat2: return 1
    return 6371 * (acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2)))

def rbf(d, h = 100):
    """Radial base function / Guassian kernel.
    
    Args:
        d (float): Distance.
        h (float, optional): Kernel width, by default 100.
    Returns:
        (float) Kernel score.
    """
    return np.exp(-d**2 / 2 / h**2)

def f_test(x, y):
    # input
    x, y = np.array(x), np.array(y)
    # F test statistic
    test_stat = np.var(x, ddof=1)/np.var(y, ddof=1)
    # degrees of freedom
    dfn, dfd = x.size - 1, y.size - 1
    # find p value of F test statistic
    pi = 1 - f.cdf(test_stat, dfn, dfd)
    return test_stat, pi