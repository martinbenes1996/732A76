# -*- coding: utf-8 -*-
"""
Run script.

@author: Martin Benes
"""

import sys
sys.path.append("src")

# plot maps
import cartography
cartography.CZ_PL_map()
cartography.SE_map()

# adjacency matrix
import plot
plot.adjacency_matrix()

# location scores
plot.adjacency_heatmap() # adjacency
plot.centroid_distance_heatmap(h = 120) # centroid distance
plot.location_score_heatmap(h = 120) # combination

# area population
plot.area_population_scatter()
plot.popdensity_boxplot_noPRG()



# run tests
import hypotheses
print( hypotheses.regions_normally_distributed(pi = False, alpha = .05) )
print( hypotheses.administrative_divisions_similar(pi = False, alpha = .05) )

# log on
import logging
logging.basicConfig(level = logging.INFO)

# DTW matching
#for h in [.01,.012,.014,.016,.018,.02]:
#    plot.deaths_dtw_czekanowski(h = h)
#    plot.deaths_dtw_heatmap(h = h)

plot.flush_cache(12345)
plot.deaths_dtw_czekanowski(h = .055, coef = 400)
 