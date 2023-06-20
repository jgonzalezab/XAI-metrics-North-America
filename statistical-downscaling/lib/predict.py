import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump, load
import rpy2.robjects as robjects

DATA_PATH_SD = './data/'

def predictTrain(modelPath, dataPath):
    
    # Load model
    models = load(modelPath + 'modelsLR.joblib')

    # Load xyT data
    robjects.r['load'](DATA_PATH_SD + 'xyT_DeepESD_train.rda')
    xyT = robjects.r['xyT_train']
    
    xData = np.array(xyT[0][0])
    xData = xData.reshape(xData.shape[0],
                          xData.shape[1] * xData.shape[2] * xData.shape[3])

    # Compute predictions
    yPred = models.predict(xData)

    # Save as npy
    np.save(DATA_PATH_SD + 'predLR_train.npy', yPred)

def predictTest(modelPath, dataPath):
    
    # Load model
    models = load(modelPath + 'modelsLR.joblib')

    # Load xyT data
    robjects.r['load'](DATA_PATH_SD + 'xyT_DeepESD_test.rda')
    xyT = robjects.r['xyT_test']
    
    xData = np.array(xyT[0][0])
    xData = xData.reshape(xData.shape[0],
                          xData.shape[1] * xData.shape[2] * xData.shape[3])

    # Compute predictions
    yPred = models.predict(xData)

    # Save as npy
    np.save(DATA_PATH_SD + 'predLR_test.npy', yPred)

def predict_GCMHist(modelPath, dataPath):
    
    # Load model
    models = load(modelPath + 'modelsLR.joblib')

    # Load xyT data
    robjects.r['load'](DATA_PATH_SD + 'xyT_DeepESD_GCM_Hist.rda')
    xyT = robjects.r['xyT_GCM_Hist']
    
    xData = np.array(xyT[0][0])
    xData = xData.reshape(xData.shape[0],
                          xData.shape[1] * xData.shape[2] * xData.shape[3])

    # Compute predictions
    yPred = models.predict(xData)

    # Save as npy
    np.save(DATA_PATH_SD + 'predLR_GCM_Hist.npy', yPred)

def predict_GCMFut(modelPath, dataPath):
    
    # Load model
    models = load(modelPath + 'modelsLR.joblib')

    # Iterate over periods  
    yearsPeriods = ('2006_2040', '2041_2070', '2071_2100')
    for period in yearsPeriods:

        # Load xyT data
        robjects.r['load'](DATA_PATH_SD + 'xyT_DeepESD_GCM_Fut_' + period + '.rda')
        xyT = robjects.r['xyT_GCM_Fut']
        
        xData = np.array(xyT[0][0])
        xData = xData.reshape(xData.shape[0],
                            xData.shape[1] * xData.shape[2] * xData.shape[3])

        # Compute predictions
        yPred = models.predict(xData)

        # Save as npy
        np.save(DATA_PATH_SD + 'predLR_GCM_Fut_' + period + '.npy', yPred)