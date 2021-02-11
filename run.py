# -*- coding: utf-8 -*-
"""
Run script.

@author: Martin Benes
"""

import sys
sys.path.append("src")
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,10)
import logging
logging.basicConfig(level = logging.INFO)

# plot maps
import cartography
cartography.CZ_PL_map()
cartography.SE_map()

# adjacency matrix
import plot
plot.adjacency_matrix()

# regional traceplots, sex/gender violin plots
plot.gender_age_violinplot()

# knn filter
plot.deaths_smooth('PL22')
plot.dtw_plot('PL9','PL22',font_size = 12)
# appendix
plot.dtw_plot_appendix()

# run tests
import hypotheses
print( hypotheses.regions_t_distributed(K = 1000, pi = True, alpha = .05) )
print( hypotheses.regions_normally_distributed(pi = True, alpha = .05) )
print( hypotheses.administrative_divisions_similar(pi = False, alpha = .05) )
# test visualization
plot.area_population_scatter()
plot.popdensity_boxplot_noPRG()
# outliers
hypotheses.IQR()

# DTW matching
plot.flush_cache(12345)
plot.deaths_dtw_czekanowski_global()
plot.deaths_dtw_czekanowski_firstWave()

plot.deaths_dtw_czekanowski_sweden()
plot.deaths_dtw_czekanowski_sweden_firstWave()

# location scores
plot.adjacency_heatmap() # adjacency
plot.centroid_distance_heatmap(h = 100) # centroid distance
plot.location_score_heatmap(h = 100) # combination

# country series
plot.deaths_country_series()

# weekday ratio
print( hypotheses.weekdays_equal_ratio(pi = False) )
#plot.weekday_ratio_heatmap()

# results
cartography.map_results(['SE121','SE312','SE313','SE125','SE231'])
cartography.map_results(['SE224','SE221','SE213','SE322','SE214'])
cartography.map_results(['CZ020','CZ031','CZ052','CZ051','CZ064',
                         'PL42$','PL63$','PL61$'])




 