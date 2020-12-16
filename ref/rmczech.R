library(RMaCzek)
x = read.csv("czekanowski.py", encoding = "UTF-8")

library(tidyr)
d <- x %>%
  dplyr::mutate(Distance = as.numeric(Distance)) %>%
  tidyr::pivot_wider(id_cols = "x", names_from = "y",
                     values_from = "Distance") %>%
  as.data.frame()
rownames(d) <- d[,1]
d <- d[,-1] %>%
  as.matrix() %>%
  as.dist()
d

M <- czek_matrix(d, interval_breaks = c(13))
print(M)
