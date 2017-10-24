Set up stacking fold.
```
Rscript set_up_stacking_folds.R
```

Fit hyperparameters.
```
python fit_hyperparams.py xgboost
python fit_hyperparams.py nn
```

Train each model for each fold.
```
python train.py config.yaml xgboost 0
...
python train.py config.yaml xgboost 4
python train.py config.yaml nn 0
...
python train.py config.yaml nn 4
```

Stack.
```
python stack.py config.yaml 
```
