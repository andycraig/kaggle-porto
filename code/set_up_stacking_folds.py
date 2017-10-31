import shutil
import pandas as pd
import numpy as np

np.random.seed(1031) # Month, date on which file was created.

n_folds = 5

train = pd.read_csv("../data/train.csv")

n_rows = len(train) 

# fold: close-to-equal numbers of each fold, in random order.
fold_vals = (np.arange(0, n_folds)
                   .repeat(np.ceil(n_rows / n_folds))
                   .shuffle()[0:n_rows])

(train.assign(fold=fold_vals)
 .drop('id') # Get rid of this so it doesn't get used accidentally for training.
 .to_csv("../generated-files/train.csv", index=False))

# Test file is as-is.
shutil.copy("../data/test.csv", "../generated-files/test.csv")
