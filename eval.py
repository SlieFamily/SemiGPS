from sklearn.metrics import (mean_squared_error, 
                             mean_absolute_error,
                             r2_score)

import numpy as np
import pandas as pd

gdp = pd.read_csv("datasets/gba_3gdp.csv", index_col=0)
y = gdp[["_1stIn", "_2ndIn", "_3rdIn"]].values
y_pred = np.load('datasets/pred_gdp.npy')

for i in range(3):
    print(f'{i+1}(th) In.')
    print('MSE: ', mean_squared_error(y[:,i], y_pred[:,i]))
    print('MAE: ', mean_absolute_error(y[:,i], y_pred[:,i]))
    print('R2: ', r2_score(y[:,i], y_pred[:,i]))
    print()