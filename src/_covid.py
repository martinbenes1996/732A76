# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 09:57:23 2020

@author: martin
"""

import sys
sys.path.append('src')

from dtw import *
import numpy as np
import pandas as pd
from scipy.spatial.distance import correlation
from sklearn.neighbors import KNeighborsRegressor

import _czekanowski
import _src

import logging
LOG = logging.getLogger(__name__)

cachedata = None
def deaths_df():
    # simple runtime caching
    global cachedata
    if cachedata is not None: return cachedata
    
    # fetch data
    cz = _src.czechia()
    pl = _src.poland()

    # rename column
    pl = pl\
        .rename({'NUTS2': 'region', 'NUTS3': 'district'}, axis = 1)

    # match same dates
    dt_range = pd.date_range(
        start = max(cz.date.min(), pl.date.min()),
        end = min(cz.date.max(), pl.date.max())
    )
    cz = cz[cz.date.isin(dt_range)]
    pl = pl[pl.date.isin(dt_range)]

    # merge into single data frame
    df = pd.concat([cz, pl])\
        .groupby(['date','week','region'])\
        .sum()\
        .reset_index()
    # fill missing
    missing_rows = {'date': [], 'week': [], 'region': [], 'deaths': []}
    for dt in dt_range:
        for reg in df.region.unique():
            row = df[(df.date == dt) & (df.region == reg)]
            if row.empty:
                missing_rows['date'].append(dt)
                missing_rows['week'].append(dt.isocalendar()[1])
                missing_rows['region'].append(reg)
                missing_rows['deaths'].append(0)
    df_full = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index = True)
    # merge mazowieckie
    is_mazowieckie = df_full.region.isin(['PL91','PL92'])
    df_mazowieckie = df_full[is_mazowieckie]\
        .groupby(['date','week'])\
            .aggregate({'deaths': 'sum'})\
                .reset_index()
    df_mazowieckie['region'] = 'PL9'
    
    # final data
    x = pd.concat([df_full[~is_mazowieckie],df_mazowieckie], ignore_index=True)\
        .sort_values(by = ['date','region'])[['date','week','region','deaths']]
    x.region = x.region.astype(str)

    # normalize by population
    pops = _src.population()
    pops.region = pops.region.astype(str)
    x = pd.merge(x, pops, on = ["region"])
    x['deaths_1K'] = x.deaths / x.population * 1000

    cachedata = x

    return x

def _to2D(ser):
    return ser.to_numpy().reshape((ser.shape[0],1))
def _predict_knn(X, y, nn = 14):
    regr = KNeighborsRegressor(n_neighbors=nn)
    regr.fit(_to2D(X), y.astype(float))
    pred = regr.predict(_to2D(X))
    return pred

def deaths_smooth(region, data = None, **kw):
    
    # fetch data if not given
    if data is None:
        data = deaths_df()
    
    # filter
    data = data[data.region == region]
    
    # smoother
    X = (data.date - data.date.min())\
        .apply(lambda i: i.days)     
    f = _predict_knn(X, data.deaths, **kw)
    
    # return x, y, fx
    return data.date.to_list(), data.deaths.to_numpy(), f

def dtw_distance(data = None, dist_method = 'euclidean'):

    # data
    x = data if data is not None else deaths_df()
    dtmin = x.date.min()
    
    # dtw distance
    regs = x.region.unique()
    reg2idx = {r:i for i,r in enumerate(regs)}
    D = np.zeros((len(regs),len(regs)))
    for name1, group1 in x.groupby('region'):
        x1 = (group1.date - dtmin).apply(lambda i: i.days)
        x2 = group1.deaths_1K
        for name2, group2 in x.groupby('region'):
            y1 = (group2.date - dtmin).apply(lambda i: i.days)
            y2 = group2.deaths_1K
            # knn smoother
            fx1 = _predict_knn(x1, x2)
            fx2 = _predict_knn(y1, y2)
            score = dtw(fx1, fx2,#group1.deaths_1K, group2.deaths_1K,
                        dist_method = dist_method)
            D[reg2idx[name1],reg2idx[name2]] = score.normalizedDistance

    # scale to [0,1]
    D /= D.max()
    
    # return
    return D

def czekanowski_dtw(data = None, dist_method = 'euclidean', **kw):

    # default configuration
    kw = {
        # kernel
        'h': .06, 'coef': 300, 'cutoff': .2,
        # GA
        'popsize': 30, 'maxiter': 1000, 'mutprob': .9, 'random_starts': 3,
        # diagram
        'diagonal': True,
        **kw
    }
    
    # data
    LOG.info("fetching data")
    D = dtw_distance(data = data, dist_method = dist_method)
    
    # columns
    regdetails = {k:v for k,v in _src.regions().items() if k[:2] in {'CZ','PL'}}
    regnames = np.array([regdetails[n]['name'] for n in regdetails])
    
    # kernel projection
    LOG.info("computing distance kernel")
    D_ = _czekanowski.distance_rbf(
        D,
        **kw
    )
    # Czekanowski diagram
    np.random.seed = 12345
    P = _czekanowski.plot(
        D_,
        cols = regnames,
        **kw
    )
    return P
