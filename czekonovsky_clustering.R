# set up data

all_four_distances <- read.csv(
  "./data/clustering_distance_datasets/all_four_distances.csv",
  row.names = 1)

all_except_euclidean_geog <- read.csv(
  "./data/clustering_distance_datasets/all_except_euclidean_geog.csv",
  row.names = 1)

all_except_adj <- read.csv(
  "./data/clustering_distance_datasets/all_except_adj.csv",
  row.names = 1)

pop_pop_dens <- read.csv(
  "./data/clustering_distance_datasets/pop_pop_dens.csv",
  row.names = 1)

pop_dens <- read.csv(
  "./data/clustering_distance_datasets/pop_dens.csv",
  row.names = 1)


# import library
library(RMaCzek)

# visualise
plot(czek_matrix(all_four_distances, scale_data = FALSE, n_classes = 6))
plot(czek_matrix(all_except_euclidean_geog, scale_data = FALSE, n_classes = 6))
plot(czek_matrix(all_except_adj, scale_data = FALSE, n_classes = 6))
plot(czek_matrix(pop_pop_dens, scale_data = FALSE, n_classes = 6))
plot(czek_matrix(pop_dens, scale_data = FALSE, n_classes = 6))


