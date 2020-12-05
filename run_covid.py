
import numpy as np
import pandas as pd

import src

# download data
cz = src.czechia()
pl = src.poland()
pl['region'] = pl.NUTS2

# population
import eurostat_deaths
pops = eurostat_deaths.populations()
pops = pops[['sex','age','geo\\time','2019']]
pops.columns = ['sex','age_group','region','population']
pops = pops[pops.region.isin(set([*pl.region.unique(), *cz.region.unique()]))]
# summarize
pops_regwise = pops[(pops.age_group == "TOTAL") & (pops.sex == "T")][['region','population']]
pops_sexwise = pops[(pops.age_group == "TOTAL") & (pops.sex != "T")][['region','sex','population']]
pops_agewise = pops[(pops.age_group != "TOTAL") & (pops.sex == "T")][['region','age_group','population']]

# match same dates
dt = pd.date_range(start = max(cz.date.min(), pl.date.min()),
                   end = min(cz.date.max(), pl.date.max()))
cz = cz[cz.date.isin(dt)]
pl = pl[pl.date.isin(dt)]

# parse
cz_regwise = cz\
    .groupby(['date','week','region'])\
    .sum()\
    .reset_index()

# join with populations
cz_regwise = cz_regwise.merge(pops_regwise, on = 'region')
cz_groupwise = cz.merge(pops, on = ['region','age_group','sex'])
cz_groupwise = cz_groupwise.merge(pops_regwise, on = ['region'])
cz_groupwise.columns = [*cz_groupwise.columns[:-2], "population","population_all"]
#cz_sexwise   = cz.merge(pops_sexwise, on = ['region','sex'])
#cz_agewise   = cz.merge(pops_agewise, on = ['region','age_group'])

# normalize deaths
def normalize_deaths(x, relpop = "population", per = 1000):
    return x.deaths * per / x[relpop]

cz_regwise['deaths_perall'] = normalize_deaths(cz_regwise)
cz_groupwise['deaths_pergroup'] = normalize_deaths(cz_groupwise)
cz_groupwise['deaths_perall'] = normalize_deaths(cz_groupwise, "population_all")
#cz_sexwise['deaths_pergroup'] = normalize_deaths(cz_regwise)
#cz_sexwise['deaths_perall'] = normalize_deaths(cz_regwise, cz)
#cz_agewise['deaths_pergroup'] = normalize_deaths(cz_regwise)
#cz_agewise['deaths_perall'] = normalize_deaths(cz_regwise, cz)

# to numpy
cz_regwise = cz_regwise\
    .pivot(index = 'date', columns = 'region', values = 'deaths_perall')\
    .fillna(0)\
    .transpose()
cz_regwise_M = cz_regwise.to_numpy()

# distance matrix
from scipy.spatial.distance import cdist, chebyshev, euclidean
cz_regwise_dist = cdist(cz_regwise_M, cz_regwise_M, metric = euclidean)

if False:
    import seaborn as sns; sns.set_theme()
    import matplotlib.pyplot as plt
    sns.heatmap(cz_regwise_dist)
    plt.show()

# seriate
from seriate import seriate
cz_regwise_order = seriate(cz_regwise_dist)
# rearrange
cz_regwise_dist_ = cz_regwise_dist[:,cz_regwise_order]
cz_regwise_dist_ = cz_regwise_dist_[cz_regwise_order,:]

if True:
    import plotly.express as px
    import matplotlib.pyplot as plt
    fig = px.imshow(cz_regwise_dist_,
                    labels = dict(x="Regions", y="Regions", color="Deaths per 1000 people"),
                    x = cz_regwise.index[cz_regwise_order], y = cz_regwise.index[cz_regwise_order])
    fig.update_xaxes(side="top")
    fig.show()

#import seaborn as sns; sns.set_theme()
#import matplotlib.pyplot as plt
#sns.heatmap(cz_regwise_dist_)
