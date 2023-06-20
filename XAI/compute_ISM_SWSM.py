import os
import re
import time
import sys
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
import innvestigate
import numpy as np
import keras
import matplotlib.pyplot as plt
import lib.models as models
import lib.utils as utils
import xarray as xr
import warnings
warnings.filterwarnings("ignore")

# PATHS
DATA_PATH = '../preprocessData/data/'
MODELS_PATH = '../statistical-downscaling/models/'
SD_PATH = '../statistical-downscaling/data/'

# Allow memory growth on GPUs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Model to compute saliency maps from
modelName = sys.argv[1]

# Day to compute saliency maps from
month = str(sys.argv[2])
year = str(sys.argv[3])

# Map to the proper topology
if 'DeepESD' in modelName:
    modelTopology = 'DeepESD'
elif 'CNNPan' in modelName:
    modelTopology = 'CNNPan'
elif 'CNN_UNET' in modelName:
    modelTopology = 'CNN_UNET'
else:
    raise ValueError('Please provide a valid model')

# Load train data from xyT objects generated during training
base = importr('base')
base.load(SD_PATH + 'xyT_' + modelTopology + '.rda')

x = np.array(robjects.r['xyT'].rx2('x.global'))
y = np.array(robjects.r['xyT'].rx2('y').rx2('Data'))

if 'CNN_UNET' in modelName:
    neuronsIdx = utils.getLandNeurons_UNET()
else:
    neuronsIdx = utils.getLandNeurons_CNNs()

# Load model and weights
modelObj = models.load_model(model = modelTopology,
                             input_shape = x.shape[1:],
                             output_shape = y.shape[1])

print('Loading {} weights into {} topology'.format(modelName, modelTopology))
modelObj.load_weights(os.path.expanduser(MODELS_PATH + modelName + '.h5'))

# Compute XAI metrics for all the gridpoints (neurons) for an specific day
print('Computing Integrated Saliency Map (ISM)')
utils.computeISM(modelObj=modelObj, modelName=modelName,
                 xData=x, neuronsIdx=neuronsIdx,
                 month=month, year=year)

print('Computing Spatially Weighted Saliency Map (SWSM)')
utils.computeSWSM(modelObj=modelObj, modelName=modelName,
                  xData=x, neuronsIdx=neuronsIdx,
                  month=month, year=year)