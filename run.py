# -*- coding: utf-8 -*-
"""
Run script.

@author: Martin Benes
"""

import sys
sys.path.append("src")

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,10)

# plot maps
import cartography
cartography.CZ_PL_map()
cartography.SE_map()

# adjacency matrix
import plot
plot.adjacency_matrix()

# knn filter
plot.deaths_smooth('PL22')
plot.dtw_plot('PL9','PL22')

# log on
import logging
logging.basicConfig(level = logging.INFO)

#for h in [.01,.012,.014,.016,.018,.02]:
#    plot.deaths_dtw_czekanowski(h = h)
#    plot.deaths_dtw_heatmap(h = h)

# DTW matching
plot.flush_cache(12345)
plot.deaths_dtw_czekanowski(h = .05, coef = 600)

# run tests
import hypotheses
print( hypotheses.regions_t_distributed(K = 1000, pi = True, alpha = .05) )
print( hypotheses.regions_normally_distributed(pi = True, alpha = .05) )
print( hypotheses.administrative_divisions_similar(pi = False, alpha = .05) )

# scatter
plot.area_population_scatter()
plot.popdensity_boxplot_noPRG()

# location scores
plot.adjacency_heatmap() # adjacency
plot.centroid_distance_heatmap(h = 120) # centroid distance
plot.location_score_heatmap(h = 120) # combination

# area population
plot.area_population_scatter()




 