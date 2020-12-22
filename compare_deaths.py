# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 09:57:23 2020

@author: martin
"""

import numpy as np
import pandas as pd

# fetch data
import src
try: cz = src.czechia()
except: pass
pl = src.poland()

import logging
logging.basicConfig(level = logging.INFO)

# rename column
pl = pl\
    .rename({'NUTS2': 'region', 'NUTS3': 'district'}, axis = 1)

# match same dates
dt_range = pd.date_range(start = max(cz.date.min(), pl.date.min()),
                         end = min(cz.date.max(), pl.date.max()))
cz = cz[cz.date.isin(dt_range)]
pl = pl[pl.date.isin(dt_range)]

# merge into single data frame
df = pd.concat([cz, pl])\
    .groupby(['date','week','region'])\
    .sum()\
    .reset_index()

# fill missing
regions = df.region.unique()

missing_rows = {'date': [], 'week': [], 'region': [], 'deaths': []}
for dt in dt_range:
    for reg in df.region.unique():
        row = df[(df.date == dt) & (df.region == reg)]
        if row.empty:
            missing_rows['date'].append(dt)
            missing_rows['week'].append(dt.isocalendar()[1]) # todo
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

x = pd.concat([df_full[~is_mazowieckie],df_mazowieckie], ignore_index=True)\
    .sort_values(by = ['date','region'])[['date','week','region','deaths']]
x.region = x.region.astype(str)

# normalize by population
pops = src.population()
pops.region = pops.region.astype(str)
x = pd.merge(x, pops, on = ["region"])
x['deaths'] = x.deaths / x.population * 1000

# dtw distance
from dtw import *
from scipy.spatial.distance import correlation
regs = x.region.unique()
reg2idx = {r:i for i,r in enumerate(regs)}
D = np.zeros((len(regs),len(regs)))
for name1, group1 in x.groupby('region'):
    for name2, group2 in x.groupby('region'):
        score = dtw(group1.deaths, group2.deaths)
        D[reg2idx[name1],reg2idx[name2]] = score.normalizedDistance
# scale to [0,1]
D /= D.max()
#D = 1 - D

regdetails = src.regions()
regnames = np.array([regdetails[n]['name'] for n in regs])

# initial order
codes = np.array([regdetails[n]['NUTS3'] for n in regs])
init = np.argsort(codes)

# Czekanowski diagram
import czekanowski
np.random.seed = 12345

D_ = czekanowski.distance_rbf(D, h = .06,
                              coef = 300, cutoff = .2)


P = czekanowski.plot(D_, cols = regnames, diagonal = True,
                     popsize = 30, maxiter = 1000, mutprob = .9, random_starts = 3)


#D_.to_csv("czekanowski.csv", index = False)

#from seriate import seriate
#D_order = seriate(D)
#D_ = D[:,D_order]
#D_ = D_[D_order,:]

d = int(np.sqrt(P.Distance.shape[0]))
PDist_np = P.Distance.to_numpy().reshape((d,d))
lab = P.y[:d].to_list()

PDist_np = np.flip(PDist_np, axis = 0)

import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (9,7)
sns.heatmap(PDist_np, xticklabels=lab, yticklabels=list(reversed(lab)))
#sns.heatmap(D_, xticklabels=regnames, yticklabels=regnames)
plt.show()
