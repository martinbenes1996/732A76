# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 09:57:23 2020

@author: martin
"""

import sys
sys.path.append('src')

from datetime import datetime,timedelta
from dtw import *
import numpy as np
from numpy.random import multinomial
import pandas as pd
from scipy.spatial.distance import correlation
from seriate import seriate
from sklearn.neighbors import KNeighborsRegressor

import covid19czechia
import covid19poland

import _czekanowski
import _src

import logging
LOG = logging.getLogger(__name__)

def czechia_weeks():
    """Fetch Czechia data and group into weeks."""
    
    # data
    x = _src.czechia()
    
    # year
    x['year'] = x.date.apply(lambda dt: int(datetime.strftime(dt, "%Y")))
    
    # group into weeks
    return x\
        .groupby(["year","week","region"])\
        .aggregate({'deaths': 'sum'})\
        .reset_index()

def poland_weeks():
    """Fetch Poland data and group into weeks."""
    
    # data
    x = _src.poland()
    
    # year
    x['year'] = x.date.apply(lambda dt: int(datetime.strftime(dt, "%Y")))
    
    # group into weeks
    return x[['year','week','NUTS2','deaths']]\
        .rename({'NUTS2': 'region'}, axis = 1)\
        .groupby(["year","week","region"])\
        .aggregate({'deaths': 'sum'})\
        .reset_index()
    
def country_deaths(): # TO weeks
    """Fetch total country covid-19 death data."""
    def _region_to_country(x):
        return x\
            .groupby(['date'])\
            .size()\
            .reset_index(name = 'deaths')
    # Poland
    pl = covid19poland.covid_death_cases(from_github = True)
    pl = _region_to_country(pl)
    pl['country'] = 'PL'
    # Czechia
    cz = _src.czechia()
    cz = _region_to_country(cz)
    cz['country'] = 'CZ'
    # Sweden
    se = sweden_resample()
    se = se[['date','deaths']]\
        .groupby(['date'])\
        .sum()\
        .reset_index()
    se['country'] = 'SE'
    
    # match same dates
    dt_range = pd.date_range(
        start = min(cz.date.min(), pl.date.min(), se.date.min()),
        end = min(cz.date.max(), pl.date.max(), se.date.max())
    )
    cz = cz[cz.date.isin(dt_range)]
    pl = pl[pl.date.isin(dt_range)]
    se = se[se.date.isin(dt_range)]
    # merge
    df = pd.concat([pl,cz,se])
    
    # fill missing
    missing_rows = {'date': [], 'country': [], 'deaths': []}
    for dt in dt_range:
        for c in df.country.unique():
            row = df[(df.date == dt) & (df.country == c)]
            if row.empty:
                missing_rows['date'].append(dt)
                missing_rows['country'].append(c)
                missing_rows['deaths'].append(0)
    df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index = True)
    
    # return
    df = df\
        .sort_values(['date','country'])\
        .reset_index(drop = True)
    return df

def gender_age_cases(): # TO WEEKS
    """Fetch total country covid-19 death data."""
    def _region_to_country(x):
        return x\
            .groupby(['date','age','sex'])\
            .size()\
            .reset_index(name = 'deaths')
    # Poland
    pl = covid19poland.covid_death_cases(from_github = True)
    pl = _region_to_country(pl)
    pl['country'] = 'PL'
    # Czechia
    cz = _src.czechia()
    cz = _region_to_country(cz)
    cz['country'] = 'CZ'
    
    # match same dates
    dt_range = pd.date_range(
        start = min(cz.date.min(), pl.date.min()),
        end = min(cz.date.max(), pl.date.max())
    )
    cz = cz[cz.date.isin(dt_range)]
    pl = pl[pl.date.isin(dt_range)]
    # merge
    df = pd.concat([pl,cz])
    
    return df

cachedata = None
def deaths_df():
    # simple runtime caching
    global cachedata
    if cachedata is not None: return cachedata
    
    # fetch data
    cz = czechia_weeks()
    pl = poland_weeks()
    se = _src.sweden()
    
    # week-year to date
    def compute_dates(x):
        x['date'] = x.apply(lambda r: "%d %d 1" % (r.year,r.week), axis = 1)
        return x.date.apply(lambda dt: datetime.strptime(dt, "%Y %W %w"))
    cz['date'] = compute_dates(cz)
    pl['date'] = compute_dates(pl)
    se['date'] = compute_dates(se)
    
    # match same dates
    dt_range = pd.date_range(
        start = min(cz.date.min(), pl.date.min(), se.date.min()),
        end = min(cz.date.max(), pl.date.max(), se.date.max())
    )
    cz = cz[cz.date.isin(dt_range)]
    pl = pl[pl.date.isin(dt_range)]
    se = se[se.date.isin(dt_range)]

    # merge into single data frame
    df = pd.concat([cz, pl, se])\
        .groupby(['date','week','region'])\
        .sum()\
        .reset_index()
    
    years = pd.Series(dt_range).apply(lambda dt: int(datetime.strftime(dt, "%Y")))
    weeks = pd.Series(dt_range).apply(lambda dt: int(datetime.strftime(dt, "%W")))
    all_weeks = pd.DataFrame({'year': years, 'week': weeks})\
        .drop_duplicates()
    all_weeks['date'] = compute_dates(all_weeks)
    missing_rows = {'date': [], 'year': [], 'week': [], 'region': [], 'deaths': []}
    for r in all_weeks.itertuples():
        for reg in df.region.unique():
            row = df[(df.region == reg) & (df.year == r.year) & (df.week == r.week)]
            if row.empty:
                missing_rows['date'].append(r.date)
                missing_rows['year'].append(r.year)
                missing_rows['week'].append(r.week)
                missing_rows['region'].append(reg)
                missing_rows['deaths'].append(0)
    df_full = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index = True)     
    # merge mazowieckie
    is_mazowieckie = df_full.region.isin(['PL91','PL92'])
    df_mazowieckie = df_full[is_mazowieckie]\
        .groupby(['date','year','week'])\
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
        .apply(lambda i: i.days / 7)     
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
        x1 = (group1.date - dtmin).apply(lambda i: i.days / 7)
        x2 = group1.deaths_1K
        for name2, group2 in x.groupby('region'):
            y1 = (group2.date - dtmin).apply(lambda i: i.days / 7)
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

def czekanowski_dtw(data = None, dist_method = 'euclidean', seriate_method = "OLO", **kw):

    # default configuration
    kw = {
        # kernel
        'h': .06, 'coef': 300, 'cutoff': .2,
        # GA
        'popsize': 30, 'maxiter': 1000, 'mutprob': .9, 'random_starts': 3,
        # diagram
        'diagonal': False,
        **kw
    }
    
    # data
    LOG.info("fetching data")
    D = dtw_distance(data = data, dist_method = dist_method)
    
    # columns
    regdetails = {k:v for k,v in _src.regions().items()}
    regnames = np.array([regdetails[n]['name'] for n in regdetails])
    
    # Czekanowski diagram
    np.random.seed = 12345
    P = _czekanowski.plot(
        D,
        cols = regnames,
        method = seriate_method,
        **kw
    )
    return P
