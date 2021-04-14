# set up data

all_four_distances <- read.csv(
  "./data/clustering_distance_datasets/all_four_distances.csv",
  row.names = 1)

all_except_geog <- read.csv(
  "./data/clustering_distance_datasets/all_except_geog.csv",
  row.names = 1)

all_except_adj <- read.csv(
  "./data/clustering_distance_datasets/all_except_adj.csv",
  row.names = 1)

all_except_pop <- read.csv(
  "./data/clustering_distance_datasets/all_except_pop.csv",
  row.names = 1)

pop_pop_dens <- read.csv(
  "./data/clustering_distance_datasets/pop_pop_dens.csv",
  row.names = 1)

pop_dens <- read.csv(
  "./data/clustering_distance_datasets/pop_dens.csv",
  row.names = 1)

pop_dens_adj <- read.csv(
  "./data/clustering_distance_datasets/pop_dens_adj.csv",
  row.names = 1)

pop_dens_geog <- read.csv(
  "./data/clustering_distance_datasets/pop_dens_geog.csv",
  row.names = 1)

ts_cases_100k_wo_dtw <-read.csv(
  "./data/clustering_distance_datasets/ts_cases_100k_wo_dtw.csv",
  row.names = 1)

ts_deaths_100k_wo_dtw <-read.csv(
  "./data/clustering_distance_datasets/ts_deaths_100k_wo_dtw.csv",
  row.names = 1)

ts_cases_100k_w_dtw <-read.csv(
  "./data/clustering_distance_datasets/ts_cases_100k_w_dtw.csv",
  row.names = 1)

ts_deaths_100k_w_dtw <-read.csv(
  "./data/clustering_distance_datasets/ts_deaths_100k_w_dtw.csv",
  row.names = 1)

# import library
library(RMaCzek)

# visualise
plot(czek_matrix(all_four_distances, scale_data = FALSE, n_classes = 6))
plot(czek_matrix(all_except_geog, scale_data = FALSE, n_classes = 6))
plot(czek_matrix(all_except_adj, scale_data = FALSE, n_classes = 6))
plot(czek_matrix(all_except_pop, scale_data = FALSE, n_classes = 6))
plot(czek_matrix(pop_pop_dens, scale_data = FALSE, n_classes = 6))
plot(czek_matrix(pop_dens, scale_data = FALSE, n_classes = 6))
plot(czek_matrix(pop_dens_adj, scale_data = FALSE, n_classes = 6))
plot(czek_matrix(pop_dens_geog, scale_data = FALSE, n_classes = 6))

par(mar=c(8,8,4,1))
plot(czek_matrix(ts_cases_100k_wo_dtw, scale_data = FALSE, n_classes = 6))
plot(czek_matrix(ts_deaths_100k_wo_dtw, scale_data = FALSE, n_classes = 6))
plot(czek_matrix(ts_cases_100k_w_dtw, scale_data = FALSE, n_classes = 4))
plot(czek_matrix(ts_deaths_100k_w_dtw, scale_data = FALSE, n_classes = 6))
