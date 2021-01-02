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

def sweden_resample():
    """Fetch Sweden data and resample into days."""
    
    # data
    x = _src.sweden()
    
    # sample weeks into days
    d = {'date': [], 'region': [], 'region_name': [], 'week': [], 'deaths': []}
    for i in x.itertuples():
        # dates
        dt = datetime.strptime("%4d-W%02d-1" % (i.year, i.week - 1), "%Y-W%W-%w")
        dt_week = [dt + timedelta(days = j) for j in range(7)]
        # sampling
        day_deaths = multinomial(i.deaths, [1/7.]*7, size = 1)[0]

        # append
        d['date'] = [*(d['date']), *dt_week]
        d['deaths'] = [*(d['deaths']), *day_deaths]
        for _ in range(7): d['region'].append(i.region)
        for _ in range(7): d['region_name'].append(i.region_name)
        for _ in range(7): d['week'].append(i.week)
    
    # return dataframe
    return pd.DataFrame(d)\
        .sort_values(['date','region'])

def country_deaths():
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

def gender_age_cases():
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
    cz = _src.czechia()
    pl = _src.poland()
    se = sweden_resample()

    # rename column
    pl = pl\
        .rename({'NUTS2': 'region', 'NUTS3': 'district'}, axis = 1)

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
    regdetails = {k:v for k,v in _src.regions().items()}
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

def _weekday_ratio(data = None, country = None, region = None):
    
    # data
    x = data if data is not None else deaths_df()

    # country filter
    if country is not None:
        x = x[x.region.apply(lambda i: i[:2] == country)]
    
    # region filter
    if region is not None:
        x = x[x.region.apply(lambda i: i == region)]
    
    # weekday
    x['weekday'] = x.date.apply(lambda d: d.weekday() + 1)
    
    # get mean ratio
    ratio = x\
        .groupby("weekday")\
        .aggregate({'deaths': 'mean'})
    ratio['deaths'] /= ratio.sum()['deaths']
    
    # deaths per weekday ratio
    return ratio.deaths

def _weekday_data(data = None, country = None, region = None):
    
    # data
    x = data if data is not None else deaths_df()

    # country filter
    if country is not None:
        x = x[x.region.apply(lambda i: i[:2] == country)]
    
    # region filter
    if region is not None:
        x = x[x.region.apply(lambda i: i == region)]
    
    # weekday
    x['weekday'] = x.date.apply(lambda d: d.weekday() + 1)
    
    # get mean ratio
    ratio = x\
        .groupby("weekday")\
        .aggregate({'deaths': 'mean'})
    ratio['deaths'] /= ratio.sum()['deaths']
    
    # deaths per weekday ratio
    return ratio.deaths



def weekday_ratio_distance(data = None, dist_method = correlation):
    
    # data
    x = data if data is not None else deaths_df()
    
    # regions
    regs = x.region.unique()
    reg2idx = {r:i for i,r in enumerate(regs)}
    regdetails = {k:v for k,v in _src.regions().items()}
    regnames = np.array([regdetails[n]['name'] for n in regdetails])
    
    # each pair distance
    D = np.zeros((len(regs),len(regs)))   
    for name1, group1 in x.groupby('region'):
        ratio1 = _weekday_ratio(data = group1)
        for name2, group2 in x.groupby('region'):
            ratio2 = _weekday_ratio(data = group2)
           
            # distance
            ratioD = dist_method(ratio1, ratio2)
            D[reg2idx[name1],reg2idx[name2]] = ratioD

    # scale to [0,1]
    D /= D.max()

    # seriate
    D_order = seriate(D)
    D_ = D[:,D_order]
    D = D_[D_order,:]
    
    # return
    return D, regnames[D_order]

