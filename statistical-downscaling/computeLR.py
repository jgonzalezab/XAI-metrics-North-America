import numpy as np
import rpy2.robjects as robjects
import lib.train as train
import lib.predict as predict

# PATHS
DATA_PATH = '../preprocessData/data/'
DATA_PATH_SD = './data/'
MODELS_PATH = './models/'

# Preprocess data
robjects.r['load'](DATA_PATH_SD + 'xyT_DeepESD.rda')
xyT = robjects.r['xyT']

xData = np.array(xyT[1])
xDataR = xData.reshape(xData.shape[0],
                       xData.shape[1] * xData.shape[2] * xData.shape[3])

yData = np.array(xyT[0][1])

# Fit linear regression model
# train.trainLR(x=xDataR, y=yData, modelPath=MODELS_PATH)

# Compute predictions
print('Computing train predictions...')
predict.predictTrain(modelPath=MODELS_PATH,
                     dataPath=DATA_PATH_SD)

print('Computing test predictions...')
predict.predictTest(modelPath=MODELS_PATH,
                    dataPath=DATA_PATH_SD)

print('Computing GCM Historical predictions...')
predict.predict_GCMHist(modelPath=MODELS_PATH,
                        dataPath=DATA_PATH_SD)

print('Computing GCM Future predictions...')
predict.predict_GCMFut(modelPath=MODELS_PATH,
                       dataPath=DATA_PATH_SD)

# Save predictions in NetCDF
r = robjects.r
r.source('./lib/numpyToR.R')