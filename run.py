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
