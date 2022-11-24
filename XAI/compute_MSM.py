import os
import re
import time
import sys
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
import innvestigate
import numpy as np
import xarray as xr
import keras
import matplotlib.pyplot as plt
import lib.models as models
import lib.utils as utils
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

# Map to the proper topology
if 'DeepESD' in modelName:
    modelTopology = 'DeepESD'
elif 'CNNPan' in modelName:
    modelTopology = 'CNNPan'
elif 'CNN_UNET' in modelName:
    modelTopology = 'CNN_UNET'
else:
    raise ValueError('Please provide a valid model')

# Neuron from which saliency maps are computed
dimRow = int(sys.argv[2]) 
dimCol = int(sys.argv[3]) 

# Load train data from xyT objects generated during training
base = importr('base')
base.load(SD_PATH + 'xyT_' + modelTopology + '.rda')

x = np.array(robjects.r['xyT'].rx2('x.global'))
y = np.array(robjects.r['xyT'].rx2('y').rx2('Data'))

# Load model and weights
modelObj = models.load_model(model = modelTopology,
                             input_shape = x.shape[1:],
                             output_shape = y.shape[1])

print('Loading {} weights into {} topology'.format(modelName, modelTopology))
modelObj.load_weights(os.path.expanduser(MODELS_PATH + modelName + '.h5'))

# Compute Mean Saliency Map over training set
print('Computing Mean Saliency Map (MSM)')
utils.computeMSM(modelObj=modelObj, modelName=modelName,
                 dimRow=dimRow, dimCol=dimCol, xData=x)
