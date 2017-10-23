library(tidyverse)

n_folds = 5
set.seed(1023) # Month, date on which file was created.

train = read_csv("../data/train.csv")

n_rows = nrow(train) 

train %>% 
  select(-id) %>% 
  # fold: close-to-equal numbers of each fold, in random order.
  mutate(fold = rep_len(seq(1, n_folds), n_rows) %>% sample(n_rows, replace = FALSE)) %>% 
  write_csv("../generated-files/train.csv")

file.copy("../data/test.csv", "../generated-files/test.csv")
