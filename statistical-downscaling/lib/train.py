import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump, load
from time import time

def trainLR(x, y, modelPath):

    print('Computing LR models')

    # Train models
    start = time()
    models = LinearRegression(n_jobs=-1).fit(x, y)
    end = time()
    print('Elapsed time {} (mins)'.format(round(end - start) / 60))

    # Save models
    dump(models, modelPath + 'modelsLR.joblib')
