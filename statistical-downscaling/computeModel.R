# Load libraries
library(loadeR)
library(transformeR) 
library(downscaleR)
library(visualizeR)
library(climate4R.value)
library(magrittr)
library(gridExtra) 
library(RColorBrewer)
library(sp)
library(keras)
library(tensorflow)
library(downscaleR.keras)
library(loadeR.2nc)
library(reticulate)

# Load functions
source('./lib/models.R')
source('./lib/train.R')
source('./lib/predict.R')

# PATHS
DATA_PATH <- '../preprocessData/data/'
DATA_PATH_SD <- './data/'
MODELS_PATH <- './models/'

# Settings for proper GPU usage
gpu_devices = tf$config$experimental$list_physical_devices('GPU')
for (device in gpu_devices) {
    tf$config$experimental$set_memory_growth(device = device, enable = TRUE)
}

# Load X and Y
load(paste0(DATA_PATH, 'x.rda'))
load(paste0(DATA_PATH, 'y.rda'))

# Split into train and test
yearsTrain <- 1980:2002
yearsTest <- 2003:2008

xTrain <- subsetGrid(x, years = yearsTrain)
yTrain <- subsetGrid(y, years = yearsTrain)

xTest <- subsetGrid(x, years = yearsTest)
yTest <- subsetGrid(y, years = yearsTest)

# Load land mask
yMask <- loadGridData(paste0(DATA_PATH, 'yMask.nc4'), var='tas')
yMask <- yMask$Data[1, , ]
yMask <- yMask * 0 + 1

# Train and save a model on the data
modelName <- 'DeepESD'
connections <- c('conv', 'dense')

trainCNN(xTrain=xTrain, yTrain=yTrain, yMask=yMask,
         modelName=modelName,
         connections=connections)

# Compute predictions on training set
predictTrain(xTrain=xTrain,
             yTrain=yTrain,
             modelName=modelName)

# Compute predictions on test set
predictTest(xTrain=xTrain,
            yTrain=yTrain,
            xTest=xTest,
            yTest=yTest,
            modelName=modelName)

# Compute predictions on historical GCM
predictGCM_Hist(xTrain=xTrain,
                yTrain=yTrain,
                modelName=modelName)

# Compute predictions on future GCM
predictGCM_Fut(xTrain=xTrain,
                yTrain=yTrain,
                modelName=modelName)