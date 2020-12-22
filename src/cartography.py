# -*- coding: utf-8 -*-
"""
Module to generate maps of regions.

@author: Martin Benes
"""

# libraries
import csv
import io
import math
import re
import tempfile
import zipfile

# data libraries
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# constants
SCALE = 10 # 1:10M
PLOT_SIZE = 30 # cm
COLOR_DEFAULT = (.9,.9,.9)
URL = 'https://gisco-services.ec.europa.eu/distribution/v2/nuts/download/ref-nuts-2016-10m.geojson.zip'

def plot_map(assets, regions, colors = {}, font_size = 15):
    """Plots the map of regions. Currently only NUTS.
    
    Args:
        assets (list): List of ISO-3 countries to cover.
        regions (list): List of regexes matching NUTS codes.
        colors (dict): Mapping of ISO-2 to tuple of 3 (rgb fractions).
        font_size (float): Font size.
    """
    
    # set the visualization
    plt.rcParams["figure.figsize"] = (7,9)
    plt.rcParams.update({'font.size': font_size})
    def get_color(s):
        try: return colors[s]
        except: return COLOR_DEFAULT
        
    # download and extract zip
    zip_bits = requests.get(URL).content
    tmp = tempfile.TemporaryFile()
    tmp.write(zip_bits)
    tmp.seek(0)
    # load as zip
    zpf = zipfile.ZipFile(tmp)
    
    # load NUTS geojsons
    geojson_files = [f for f in zpf.namelist() if f.split('.')[-1] == 'geojson']
    geojsons = {f: geopandas.read_file(zpf.open(f)) for f in geojson_files}
    # load nuts mapper
    with io.TextIOWrapper(zpf.open('NUTS_RG_BN_10M_2016.csv', 'r'), encoding="utf-8") as fp:
        rdr = csv.reader(fp)
        _ = next(rdr) # drop header
        csv_file = list(rdr)
        nuts,code = [i[0] for i in csv_file],[int(i[1]) for i in csv_file]
        nutsmap = pd.DataFrame({'NUTS_BN_ID': code, 'id': nuts})

    # filters
    match_region = lambda s: s is not None and any(re.match(r,s) for r in regions)
    def keep_level_bn(lvl):
        def _drop_out_level(s):
            r = "^[A-Z]{2}[0-9]{%d}$" % lvl
            return s is not None and bool(re.match(r,s))
        return _drop_out_level

    # parse relevant files
    boundaries,centers,shapes = {},{},{}
    for lvl in range(4):
        # corresponding file per level
        bn_key = f'NUTS_BN_{str(SCALE).zfill(2)}M_2016_4326_LEVL_{lvl}.geojson'
        lb_key = f'NUTS_LB_2016_4326_LEVL_{lvl}.geojson'
        rg_key = f'NUTS_RG_{str(SCALE).zfill(2)}M_2016_4326_LEVL_{lvl}.geojson'
        # get boundaries
        boundaries[lvl] = geojsons[bn_key]
        centers[lvl]    = geojsons[lb_key]
        shapes[lvl]     = geojsons[rg_key]
        # map boundary ID onto NUTS ID
        boundaries[lvl] = boundaries[lvl]\
            .merge(nutsmap, how = 'inner', on = 'NUTS_BN_ID')
        # filter boundaries from different level
        boundaries[lvl] = boundaries[lvl][boundaries[lvl].id.apply(keep_level_bn(lvl))]
        centers[lvl] = centers[lvl][centers[lvl].id.apply(keep_level_bn(lvl))]
        # keep only input
        centers[lvl] = centers[lvl][centers[lvl].id.apply(match_region)]
        shapes[lvl] = shapes[lvl][shapes[lvl].id.apply(match_region)]
        boundaries[lvl] = boundaries[lvl][boundaries[lvl].id.apply(match_region)]

    # concat the levels
    boundaries = pd.concat(boundaries.values(), ignore_index = True)
    centers = pd.concat(centers.values(), ignore_index = True)
    shapes = pd.concat(shapes.values(), ignore_index = True)

    # keep only longest centers
    to_remove = []
    for i in range(int(centers.id.apply(len).max()), 2, -1):
        cs = centers[centers.id.apply(len) == i].id
        cs_ = cs.apply(lambda s: s[:-1]).unique()
        for code in cs_:
            if not centers[centers.id == code].empty:
                to_remove.append(code)
    centers = centers[~centers.id.isin(to_remove)]

    # load library data
    borders = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    borders = borders[~borders.iso_a3.isin(assets)]

    # ===== plot map =====
    # aspect ratio
    minx, miny, maxx, maxy = shapes.total_bounds
    w = 1 / math.cos(math.radians((maxy+miny)/2 - 5))
    # prepare plot
    fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE / w), facecolor=(1,1,1))
    ax.axis('off') # remove axis

    # plot continent shades
    borders.plot(ax = ax, color = (.97,.97,.97), linewidth = .5)
    # plot shapes
    for name,group in shapes.groupby('CNTR_CODE'):
        group.plot(ax = ax, color = get_color(name))
    # add boundaries
    boundaries.plot(ax = ax, color = "black", linewidth = .8, aspect = w)
    # add labels
    centers.apply(lambda s: plt.annotate(text = s.id, xy = s.geometry.coords[0], ha='center'),axis=1)

    # set crop
    x_over,y_over = min(1, 20/(maxx - minx)),min(1, 20/(maxy - minx))
    ax.set_xlim(minx - x_over, maxx + x_over)
    ax.set_ylim(miny - y_over, maxy + y_over)
    plt.show()


def CZ_PL_map():
    """Constructs map of Czechia and Poland."""
    plot_map(
        ['CZ', 'CZE', 'PL', 'POL'],
        [r'CZ[0-9]{0,3}$', r'PL$', r'PL9$', r'PL[^9][0-9]{0,1}$'],
        colors = {'CZ': (1,.91,.51), 'PL': (1,.99,.78)},
        font_size = 20
    )

def SE_map():
    """Constructs map of Sweden."""
    plot_map(
        ['SE', 'SWE'],
        [r'SE[0-9]*'],
        colors = {'SE': (.96,.835,.28)},
        font_size = 12
    )
    
CZ_PL_map()
SE_map()