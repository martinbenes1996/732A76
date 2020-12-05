
import csv
import json
import numpy as np
import pandas as pd

import covid19czechia
import covid19dh
import covid19poland
import eurostat_deaths

import tools

restriction_cols = [
    'school_closing','workplace_closing','cancel_events','gatherings_restrictions','transport_closing',
    'stay_home_restrictions','internal_movement_restrictions','testing_policy','contact_tracing'
]
def restrictions(country, level = 3):
    data,_ = covid19dh.covid19(country, level = level, verbose = False) # load data
    if level >= 1: level_cols = ['iso_alpha_2']
    if level >= 2: level_cols = [*level_cols,'administrative_area_level_2']
    if level == 3: level_cols = [*level_cols,'administrative_area_level_3']
    x = data[['date','iso_alpha_3',*level_cols,*restriction_cols]].reset_index(drop = True)
    ## restrictions imposed
    #dx = x[restriction_cols].diff()
    #x = x[dx.isin([-4.,-3.,-2.,-1.,1.,2.,3.,4.]).any(axis = 1)]
    x['week'] = x.date.apply( lambda dt: dt.isocalendar()[1] )
    x = x[["week",*level_cols,*restriction_cols]]\
        .groupby(["week",*level_cols])\
        .aggregate({c:np.nanmean for c in restriction_cols})\
        .reset_index()
    if level > 1:
        cities = x\
            .drop_duplicates(subset = ["iso_alpha_2","administrative_area_level_3"])
        #cities = cities[["iso_alpha_2","administrative_area_level_3"]]\
        #    .to_records(index=False)
        print(cities.administrative_area_level_3)
        print(cities.iso_alpha_2)
        mapcities = tools.nuts(city = cities.administrative_area_level_3, country = cities.iso_alpha_2)
        #mapcities = {city: tools.nuts(city, country) for country,city in list(cities)}
        print(mapcities)
    return x
    
def czechia():
    """Fetch Czechia data."""
    return covid19czechia.covid_deaths(level = 3, usecache = True)
def poland():
    """Fetch Poland data."""
    return covid19poland.covid_deaths(level = 3)
def population():
    """Fetch population data from 2019."""
    population = eurostat_deaths.populations()
    population = population[(population.age == "TOTAL") & (population.sex == "T")][['geo\\time','2019']]
    population.columns = ['region','population']
    return population

def regions():
    """Load regions file."""
    with open("data/regions.csv", encoding = "UTF-8") as fp:
        regions = {r['NUTS3']: r for r in csv.DictReader(fp)}
    return regions
def region_names(nuts):
    """Region names for given nuts."""
    regs = regions()
    region_names = [regs[r]['name'] for r in nuts]
    
    # return
    return region_names
def regions_df():
    # load regions
    regs = regions()
    
    # create dataframe
    regs_df = pd.DataFrame({
        'Region': [r['name'] for r in regs.values()],
        'Code': list(regs.keys()),
        'Population': [int(r['population']) for r in regs.values()],
        'Area': [float(r['area']) for r in regs.values()],
        'Country': [r[:2] for r in regs]
    })
    regs_df['Density'] = regs_df.Population / regs_df.Area
    
    # return
    return regs_df
    
def neighbors():
    """Load adjancecy file."""
    with open("data/adjacency.json") as fp:
        neighbors = json.load(fp)
        for nb in neighbors:
            neighbors[nb].append(nb)
    return neighbors

if __name__ == "__main__":
    x = restrictions("CZE")
    #print(x)