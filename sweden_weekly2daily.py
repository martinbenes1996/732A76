
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9,7)

# download data
import covid19czechia
import covid19poland
cz = covid19czechia.covid_deaths()
pl = covid19poland.covid_deaths()

# datahub
import covid19dh
dh,_ = covid19dh.covid19("United Kingdom", level = 2, verbose = False)
dh['week'] = dh.date.apply(lambda d: d.isocalendar()[1])
dh['deaths'] = dh\
    .groupby('administrative_area_level_2')['deaths']\
    .diff()\
    .fillna(0)

# weekday
pl['year'] = pl.date.apply(lambda d: d.year)
cz['year'] = pl.date.apply(lambda d: d.year)
dh['year'] = dh.date.apply(lambda d: d.year)
pl['weekday'] = pl.date.apply(lambda d: d.weekday() + 1)
cz['weekday'] = cz.date.apply(lambda d: d.weekday() + 1)
dh['weekday'] = dh.date.apply(lambda d: d.weekday() + 1)
x = dh
#x = pl.append(cz)

# create index
import pandas as pd
idx = [(y,w,wd) for y in x.year.unique()
                for w in range(x.week.min(), x.week.max() + 1)
                for wd in range(x.weekday.min(), x.weekday.max() + 1)]
# over all regions
xx = x[["year","week","weekday","deaths"]]\
    .groupby(["year","week","weekday"])\
    .sum()\
    .reindex(pd.Index(idx), fill_value = 0)\
    .reset_index()
xx.columns = ["year","week","weekday", *xx.columns[3:]]


# over everything
total = xx\
    .groupby('week')\
    .aggregate({'deaths': 'sum'})\
    .reset_index()\
    .rename(mapper = {'deaths': 'total'}, axis = 1)
perweek = xx.merge(total, on = "week")[['deaths','total']].to_numpy()
distribution_total = xx[['year','week','weekday']]
distribution_total['deaths'] = perweek[:,0] / perweek[:,1]
# average
day_overall = distribution_total\
    .groupby('weekday')\
    .aggregate({'deaths': 'mean'})


# over everything
total = xx\
    .groupby(['year','week'])\
    .aggregate({'deaths': 'sum'})\
    .reset_index()\
    .rename(mapper = {'deaths': 'total'}, axis = 1)
perweek = xx.merge(total, on = "week")[['deaths','total']].to_numpy()
distribution_total = xx[['year','week','weekday']]
perweek[:,1][perweek[:,1] == 0] = 1 # for zero-division
distribution_total['deaths'] = perweek[:,0] / perweek[:,1]
# average
distribution_total['year_week'] = distribution_total.year.astype(str)+"_"+distribution_total.week.astype(str)
perweek_overall = distribution_total\
    .pivot(index = 'weekday', columns = 'year_week', values = "deaths")
P = perweek_overall\
    .to_numpy()
# t-test
import numpy as np
from scipy.stats import t    
tt = (P.mean(axis = 1) - 1/7) / np.sqrt(P.var(axis = 1) / P.shape[1])
#t_ci = t.interval(.05, P.shape[1]-1, P.mean(axis=1), P.std(axis=1))
pval = t.sf(np.abs(tt), P.shape[1]-1)*2
pval > 0.05

fig, ax = plt.subplots()
days = ["Mo","Tu","We","Th","Fr","Sa","Su"]
ptt = pd.DataFrame(P.mean(axis = 1), index = days, columns = ["per day"])\
    .plot(kind = "bar", ax = ax)
ax.axhline(1/7, color = "red", label = "uniform")
ax.legend()
plt.show()

fig, ax = plt.subplots()
pd.DataFrame(pval, index = days, columns = ["pi"])\
    .plot(kind = "bar", ax = ax)
ax.axhline(0.05, color = "tomato", label = "alpha")
ax.legend()
plt.yscale("log")
plt.show()

day_overall = distribution_total\
    .groupby('year','week','weekday')\
    .aggregate({'deaths': 'mean'})
