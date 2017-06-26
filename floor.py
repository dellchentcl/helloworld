import numpy as np
import pandas as pd
import sklearn.linear_model as linear

DISTANCE_NEAR       = 2
DISTANCE_FAR        = 50
MANUALLY_ADJUSTMENT = 0.063

def floor(dataset):
    global DISTANCE_NEAR, DISTANCE_FAR

    ds = dataset[:, [2,3]]

    # use 30m~150m ground data
    ds = ds[ds[:,0] > DISTANCE_NEAR]
    ds = ds[ds[:,0] < DISTANCE_FAR]
    ds = ds[ds[:,1] < 0]

    ''' linear regression for coef_ '''
    model = linear.LinearRegression()
    model.fit(ds[:, 0].reshape(-1,1), ds[:, 1].reshape(-1,1))
    print("coef_: ", model.coef_)

    dataset[:, 3] -= dataset[:, 2] * (model.coef_ - MANUALLY_ADJUSTMENT).squeeze()
    return dataset