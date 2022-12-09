import os
import time
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import keras
import innvestigate
import xarray as xr
import pandas as pd
import datetime as dt
import glob
from tqdm import tqdm

# PATHS
XAI_PATH = './data/'
DATA_PATH = '../preprocessData/data/'
SD_PATH = '../statistical-downscaling/data/'

# Get the proper neuron index for DeepESD and CNNPan
def getNeuronIndex_CNNs(dimRow, dimCol):
    
    # Load unflattened Y data
    base = importr('base')
    base.load(SD_PATH + 'xyT_CNN_UNET.rda') # We need the unflattened data
    yRef = np.array(robjects.r['xyT'].rx2('y').rx2('Data'))[0, :, :]

    # Load land mask
    yMask = xr.open_dataset(DATA_PATH + 'yMask.nc4')
    yMask = yMask['tas'].values[0, :, :]
    yMask = yMask * 0 + 1

    # Fill with CNNs neurons position
    neuronsArray = np.empty_like(yRef)
    neuronsArray = neuronsArray * yMask

    nCols = neuronsArray.shape[1]
    initValue = 0
    for col in range(nCols):
        notNan = (~np.isnan(neuronsArray[:, col]))
        neuronsArray[:, col][notNan] = list(range(initValue, initValue +( np.sum(notNan))))
        initValue = initValue + (np.sum(notNan))

    # Check for errors in the input
    if (dimRow >= neuronsArray.shape[0]) or (dimCol >= neuronsArray.shape[1]):
        raise ValueError('Please provide a dimRow and dimCol value within the map bounds')

    neuronIdx = neuronsArray[dimRow, dimCol]
    if np.isnan(neuronIdx):
        raise ValueError('Please provide a point within the land area')

    return int(neuronIdx)

# Get the neurons index for land gridpoints of DeepESD and CNN-PAN models
def getLandNeurons_CNNs():

    # Load unflattened Y data
    base = importr('base')
    base.load(SD_PATH + 'xyT_CNN_UNET.rda') # We need the unflattened data
    yRef = np.array(robjects.r['xyT'].rx2('y').rx2('Data'))[0, :, :]

    # Load land mask
    yMask = xr.open_dataset(DATA_PATH + 'yMask.nc4')
    yMask = yMask['tas'].values[0, :, :]
    yMask = yMask * 0 + 1

    # Load region mask data
    yMask_RegionSea = np.load(DATA_PATH + 'maskRegionsSea.npy')

    # Fill with CNNs neurons position
    neuronsArray = np.empty_like(yRef)
    neuronsArray = neuronsArray * yMask

    nCols = neuronsArray.shape[1]
    initValue = 0
    for col in range(nCols):
        notNan = (~np.isnan(neuronsArray[:, col]))
        neuronsArray[:, col][notNan] = np.array(range(initValue, initValue + (np.sum(notNan))))
        initValue = initValue + (np.sum(notNan))

    # Return land indices
    neuronsArray = neuronsArray * yMask_RegionSea
    neuronsArray = neuronsArray.flatten()
    neuronsArray = neuronsArray[~np.isnan(neuronsArray)]
    neuronsArray = neuronsArray.tolist()
    neuronsArray = [int(elem) for elem in neuronsArray]

    return neuronsArray

# Insert a list of values following the order of the neurons (CNNs models)
def insertValues_CNNs(valuesToInsert):

    # Load unflattened Y data
    base = importr('base')
    base.load(SD_PATH + 'xyT_CNN_UNET.rda') # We need the unflattened data
    yRef = np.array(robjects.r['xyT'].rx2('y').rx2('Data'))[0, :, :]

    # Load land mask
    yMask = xr.open_dataset(DATA_PATH + 'yMask.nc4')
    yMask = yMask['tas'].values[0, :, :]
    yMask = yMask * 0 + 1

    # Load region mask data
    yMask_RegionSea = np.load(DATA_PATH + 'maskRegionsSea.npy')

    # Fill with CNNs neurons position
    neuronsArray = np.empty_like(yRef)
    neuronsArray = neuronsArray * yMask

    nCols = neuronsArray.shape[1]
    initValue = 0
    for col in range(nCols):
        notNan = (~np.isnan(neuronsArray[:, col]))
        neuronsArray[:, col][notNan] = valuesToInsert[initValue:(initValue + (np.sum(notNan)))]
        initValue = initValue + (np.sum(notNan))

    neuronsArray = neuronsArray * yMask_RegionSea

    return neuronsArray

# Save predictand neurons indices for SWSM computation (for CNNs)
def saveNeuronsIndexArray_CNNs():

     # Load unflattened Y data
    base = importr('base')
    base.load(SD_PATH + 'xyT_CNN_UNET.rda') # We need the unflattened data
    yRef = np.array(robjects.r['xyT'].rx2('y').rx2('Data'))[0, :, :]

    # Load land mask
    yMask = xr.open_dataset(DATA_PATH + 'yMask.nc4')
    yMask = yMask['tas'].values[0, :, :]
    yMask = yMask * 0 + 1

    # Fill with CNNs neurons position
    neuronsArray = np.empty_like(yRef)
    neuronsArray = neuronsArray * yMask

    nCols = neuronsArray.shape[1]
    initValue = 0
    for col in range(nCols):
        notNan = (~np.isnan(neuronsArray[:, col]))
        neuronsArray[:, col][notNan] = list(range(initValue, initValue +( np.sum(notNan))))
        initValue = initValue + (np.sum(notNan))

    base.load(DATA_PATH + 'x.rda')
    lats = np.array(robjects.r['x'].rx2('xyCoords').rx2('y'))
    lons = np.array(robjects.r['x'].rx2('xyCoords').rx2('x'))

    lats = np.expand_dims(lats, axis=1)
    lats = np.repeat(lats, repeats = lons.shape, axis=1)

    lons = np.expand_dims(lons, axis=0)
    lons = np.repeat(lons, repeats = lats.shape[0], axis=0)

    coordsX = np.stack((lats, lons))
    np.save(XAI_PATH + 'coordsX.npy', coordsX)

    base.load(DATA_PATH + 'y.rda')
    lats = np.array(robjects.r['y'].rx2('xyCoords').rx2('y'))
    lons = np.array(robjects.r['y'].rx2('xyCoords').rx2('x'))

    lats = np.expand_dims(lats, axis=1)
    lats = np.repeat(lats, repeats = lons.shape, axis=1)

    lons = np.expand_dims(lons, axis=0)
    lons = np.repeat(lons, repeats = lats.shape[0], axis=0)

    coordsY = np.stack((lats, lons))

    neuronsArray = np.expand_dims(neuronsArray, axis=0)
    finalArray = np.concatenate((neuronsArray, coordsY), axis=0)
    np.save(XAI_PATH + 'neuronsPalette_CNNs.npy', finalArray)

# Get the proper neuron index for CNN_UNET
def getNeuronIndex_CNN_UNET(dimRow, dimCol):
    
    # Load Y data
    base = importr('base')
    base.load(SD_PATH + 'xyT_CNN_UNET.rda')
    yRef = np.array(robjects.r['xyT'].rx2('y').rx2('Data'))[0, :, :]

    # Load land mask
    yMask = xr.open_dataset(DATA_PATH + 'yMask.nc4')
    yMask = yMask['tas'].values[0, :, :]
    yMask = yMask * 0 + 1

    # Fill with CNNs neurons position
    neuronsArray = np.empty_like(yRef)

    nCols = neuronsArray.shape[1]
    for row in range(neuronsArray.shape[0]):
        neuronsArray[row, :] = np.array(range(nCols - neuronsArray.shape[1], nCols))
        nCols = nCols + neuronsArray.shape[1]

    neuronsArray = neuronsArray * yMask

    # Check for errors in the input
    if (dimRow >= neuronsArray.shape[0]) or (dimCol >= neuronsArray.shape[1]):
        raise ValueError('Please provide a dimRow and dimCol value within the map bounds')

    neuronIdx = neuronsArray[dimRow, dimCol]
    if np.isnan(neuronIdx):
        raise ValueError('Please provide a point within the land area')

    return int(neuronIdx)

# Get the neurons index for land gridpoints of CNN-UNET model
def getLandNeurons_UNET():
    
    # Load Y data
    base = importr('base')
    base.load(SD_PATH + 'xyT_CNN_UNET.rda')
    yRef = np.array(robjects.r['xyT'].rx2('y').rx2('Data'))[0, :, :]

    # Get neurons indices corresponding to land and proper regions
    yMask_RegionSea = np.load(DATA_PATH + 'maskRegionsSea.npy')

    # Initialize array of neuron indices
    neuronsArray = np.empty_like(yRef)
    
    # Fill with neuron indices
    nCols = neuronsArray.shape[1]
    for row in range(neuronsArray.shape[0]):
        neuronsArray[row, :] = np.array(range(nCols - neuronsArray.shape[1], nCols))
        nCols = nCols + neuronsArray.shape[1]

    # Return land indices
    neuronsArray = neuronsArray * yMask_RegionSea
    neuronsArray = neuronsArray[~np.isnan(neuronsArray)]
    neuronsArray = neuronsArray.tolist()
    neuronsArray = [int(elem) for elem in neuronsArray]

    return neuronsArray

# Insert a list of values following the order of the neurons (CNN-UNET)
def insertValues_CNN_UNET(valuesToInsert):
    
    # Load Y data
    base = importr('base')
    base.load(SD_PATH + 'xyT_CNN_UNET.rda')
    yRef = np.array(robjects.r['xyT'].rx2('y').rx2('Data'))[0, :, :]

    # Load land mask
    yMask = xr.open_dataset(DATA_PATH + 'yMask.nc4')
    yMask = yMask['tas'].values[0, :, :]
    yMask = yMask * 0 + 1

    # Fill with CNNs neurons position
    neuronsArray = np.empty_like(yRef)

    nCols = neuronsArray.shape[1]
    for row in range(neuronsArray.shape[0]):
        neuronsArray[row, :] = valuesToInsert[(nCols - neuronsArray.shape[1]):nCols]
        nCols = nCols + neuronsArray.shape[1]

    neuronsArray = neuronsArray * yMask

    return neuronsArray

# Save predictand neurons indices for SWSM computation (for CNN-UNET)
def saveNeuronsIndexArray_CNN_UNET():
    
    # Load Y data
    base = importr('base')
    base.load(SD_PATH + 'xyT_CNN_UNET.rda')
    yRef = np.array(robjects.r['xyT'].rx2('y').rx2('Data'))[0, :, :]

    # Load land mask
    yMask = xr.open_dataset(DATA_PATH + 'yMask.nc4')
    yMask = yMask['tas'].values[0, :, :]
    yMask = yMask * 0 + 1

    # Fill with CNN UNET neurons position
    neuronsArray = np.empty_like(yRef)

    nCols = neuronsArray.shape[1]
    for row in range(neuronsArray.shape[0]):
        neuronsArray[row, :] = np.array(range(nCols - neuronsArray.shape[1], nCols))
        nCols = nCols + neuronsArray.shape[1]

    neuronsArray = neuronsArray * yMask

    base.load(DATA_PATH + 'x.rda')
    lats = np.array(robjects.r['x'].rx2('xyCoords').rx2('y'))
    lons = np.array(robjects.r['x'].rx2('xyCoords').rx2('x'))

    lats = np.expand_dims(lats, axis=1)
    lats = np.repeat(lats, repeats = lons.shape, axis=1)

    lons = np.expand_dims(lons, axis=0)
    lons = np.repeat(lons, repeats = lats.shape[0], axis=0)

    coordsX = np.stack((lats, lons))
    np.save(XAI_PATH + 'coordsX.npy', coordsX)

    base.load(DATA_PATH + 'y.rda')
    lats = np.array(robjects.r['y'].rx2('xyCoords').rx2('y'))
    lons = np.array(robjects.r['y'].rx2('xyCoords').rx2('x'))

    lats = np.expand_dims(lats, axis=1)
    lats = np.repeat(lats, repeats = lons.shape, axis=1)

    lons = np.expand_dims(lons, axis=0)
    lons = np.repeat(lons, repeats = lats.shape[0], axis=0)

    coordsY = np.stack((lats, lons))

    neuronsArray = np.expand_dims(neuronsArray, axis=0)
    finalArray = np.concatenate((neuronsArray, coordsY), axis=0)
    np.save(XAI_PATH + 'neuronsPalette_CNN_UNET.npy', finalArray)

# Get X date indices for a specific date (month. year)
def getMonthIndices(month, year):

    # Returns indices for a month of a specific year
    # The way this function is possible is because the dim of
    # the X used to trained the model is the same as the one
    # used here
    # [!] This is only valid for the training set [!]

    # Load data containing the reference dates
    base = importr('base') 
    base.load('../preprocessData/data/x.rda')

    # Subset the training set
    transformeR = importr('transformeR')
    xTrain = transformeR.subsetGrid(robjects.r['x'],
                                    years=list(range(1980, 2002+1)))

    # Extract dates
    refDates = np.array(transformeR.getRefDates(xTrain)).tolist()
    refDates = [elem[:-16] for elem in refDates]

    # Get indices
    queryDate = year + '-' + month
    indices = []
    for idx, elem in enumerate(refDates):
        if elem == queryDate:
            indices.append(idx)

    # Free some memory
    robjects.r('rm(x); gc()')
    robjects.r('rm(xTrain); gc()')

    return indices

# Compute the mean saliency map over training set
def computeMSM(modelObj, modelName, dimRow, dimCol, xData):

    # Get proper neuron index w.r.t. the model
    if ('DeepESD' in modelName) or ('CNNPan' in modelName):
        neuronIdx = getNeuronIndex_CNNs(dimRow=dimRow, dimCol=dimCol)
    elif 'CNN_UNET' in modelName:
        neuronIdx = getNeuronIndex_CNN_UNET(dimRow=dimRow, dimCol=dimCol)
    else:
        raise ValueError('Please provide a valid modelName')

    # Set batch size for computing the SMs
    # Can't increase batchSize above 1 due to an existing bug in iNNvestigate
    # https://github.com/albermax/innvestigate/issues/246
    batchSize = 1

    # Pre-allocate memory for the saliencyMaps array
    saliencyMaps = np.empty(xData.shape)

    # Create analyzer
    analyzer = innvestigate.create_analyzer(name = 'integrated_gradients',
                                            model = modelObj,
                                            neuron_selection_mode = 'index')

    # First batch
    saliencyMaps[:batchSize, :] = analyzer.analyze(xData[:batchSize, :, :, :], neuronIdx)
    saliencyMaps[:batchSize, :] = np.abs(saliencyMaps[:batchSize, :])
    saliencyMaps[:batchSize, :] = saliencyMaps[:batchSize, :] / np.max(saliencyMaps[:batchSize, :])

    # Iterate over batches
    for i in tqdm(range(batchSize, xData.shape[0], batchSize)):

        # Compute raw SM
        saliencyMaps[i:i+batchSize, :] = analyzer.analyze(xData[i:i+batchSize, :, :, :], neuronIdx)
        
        # Compute absolute value and divide by the maximum value
        saliencyMaps[i:i+batchSize, :] = np.abs(saliencyMaps[i:i+batchSize, :])
        saliencyMaps[i:i+batchSize, :] = saliencyMaps[i:i+batchSize, :] / np.max(saliencyMaps[i:i+batchSize, :])

    # Filter noise
    saliencyMaps[saliencyMaps < 0.1] = 0

    # Compute mean of saliency maps
    saliencyMaps = np.mean(saliencyMaps, axis=0)

    # Save saliencyMaps as npy
    np.save(file = XAI_PATH + 'SM_MSM_' + modelName + '_dimRow' + str(dimRow) + '_dimCol' + str(dimCol) + '.npy',
            arr = saliencyMaps)

# Compute Integrated Saliency Map (ISM)
def computeISM(modelObj, modelName, xData, neuronsIdx, month, year):

    # Set batch size for computing the SMs
    # Can't increase batchSize above 1 due to an existing bug in iNNvestigate
    # https://github.com/albermax/innvestigate/issues/246
    batchSize = 1

    # Create analyzer
    analyzer = innvestigate.create_analyzer(name = 'integrated_gradients',
                                            model = modelObj,
                                            neuron_selection_mode = 'index')

    # Get month indices
    indicesMonth = getMonthIndices(year=year, month=month)

    # Pre-allocate memory for the saliencyMaps array
    saliencyMapMonth = np.empty((len(indicesMonth), xData.shape[1], xData.shape[2], xData.shape[3]))

    # Iterate over batches
    # Iterate over the full month. For each day aggregate SM across gridpoints
    # and divide by the number of gridpoints. Concatenate into saliencyMapMonth.
    # Then perform the final mean over the days of the month

    for idx, elem in tqdm(enumerate(indicesMonth), total=len(indicesMonth)):
        saliencyMapAgg = np.empty((len(neuronsIdx), xData.shape[1], xData.shape[2], xData.shape[3]))
        idxAux = 0
        for i in neuronsIdx:
            
            SMs = analyzer.analyze(np.expand_dims(xData[elem, :, :, :], axis=0), i)
            SMs = np.absolute(SMs)
            
            SMs = SMs / np.max(SMs)

            # Filter SMs
            SMs[SMs < 0.1] = 0

            saliencyMapAgg[idxAux, :, :, :] = SMs
            idxAux = idxAux + 1

        saliencyMapAgg = np.mean(saliencyMapAgg, axis=0)
        saliencyMapMonth[idx, :, :, :] = saliencyMapAgg

    ISM_metric = np.mean(saliencyMapMonth, axis=0)

    # Save saliencyMaps as npy
    np.save(file = XAI_PATH + 'SM_ISM_' + modelName + '_' + month + '_' + year + '.npy',
            arr = ISM_metric)

# Compute Spatially Weighted Saliency Map (SWSM)
def computeSWSM(modelObj, modelName, xData, neuronsIdx, month, year, var):

    # Compute index data if it is not already computed
    if not os.path.isfile(XAI_PATH + 'neuronsPalette_CNN_UNET.npy') or not os.path.isfile(XAI_PATH + 'neuronsPalette_CNNs.npy'):
        saveNeuronsIndexArray_CNNs()
        saveNeuronsIndexArray_CNN_UNET()

    # Variables list (for index purposes)
    allVars = ['z@500', 'z@700', 'z@850', 'z@1000',
               'hus@500', 'hus@700', 'hus@850', 'hus@1000',
               'ta@500', 'ta@700', 'ta@850', 'ta@1000',
               'ua@500', 'ua@700', 'ua@850', 'ua@1000',
               'va@500', 'va@700', 'va@850', 'va@1000']

    # Select the index of the provided variable
    if var not in allVars:
        raise ValueError('Please provide a valid variable')
    else:
        varIdx = allVars.index(var)

    # Iterate over all neurons to avoid issues when inserting the SWSM
    # values in the Y array
    if modelName == 'CNN_UNET':
        fullNeuronIdxs = list(range(117*211))
    else:
        fullNeuronIdxs = list(range(10870))

    # Create analyzer
    analyzer = innvestigate.create_analyzer(name = 'integrated_gradients',
                                            model = modelObj,
                                            neuron_selection_mode = 'index')

    # Get month indices
    indicesMonth = getMonthIndices(year=year, month=month)

    # Pre-allocate memory for the saliencyMaps array
    saliencyMapMonth = np.empty((len(indicesMonth), 117, 211))

    # Iterate over batches
    # First loop over the days composing the selected month. Then compute the SWSM for
    # each predicted gridpoint (output neuron) of a day. Finally average over these days, getting
    # the mean SWSM for that month
    for idx, elem in tqdm(enumerate(indicesMonth), total=len(indicesMonth)):
        metricPerNeuron = []

        for i in fullNeuronIdxs:
            if i in neuronsIdx:

                # Compute relative saliency map
                SMsMean = analyzer.analyze(np.expand_dims(xData[elem, :, :, :], axis=0), i)
                SMsMean = np.absolute(SMsMean)
                
                SMsMean = SMsMean / np.max(SMsMean)

                # Filter SMs
                SMsMean[SMsMean < 0.1] = 0

                # Select the proper variable
                SMsMean = SMsMean[0, :, :, varIdx]

                # Compute distances array (haversine distance)
                coordsX = np.load(XAI_PATH + 'coordsX.npy')
                if modelName == 'CNN_UNET':
                    arrayIndices = np.load(XAI_PATH + 'neuronsPalette_CNN_UNET.npy')
                else:
                    arrayIndices = np.load(XAI_PATH + 'neuronsPalette_CNNs.npy')

                coordsWhere = np.argwhere(arrayIndices[0, :, :] == i).tolist()[0]

                centerPoint = (int(round(coordsWhere[0]/(118/30))),
                               int(round(coordsWhere[1]/(212/53)))) 
                coordToCompare = coordsX[:, centerPoint[0], centerPoint[1]]

                coordsX = coordsX * np.pi/180
                coordToCompare = [elemC * np.pi/180 for elemC in coordToCompare]

                elem1 = np.sin((coordsX[0, :, :] - coordToCompare[0])/2)**2
                elem2 = np.cos(coordToCompare[0]) * np.cos(coordsX[0, :, :])
                elem3 = np.sin((coordsX[1, :, :] - coordToCompare[1])/2)**2
                haversineDistance = 2*np.arcsin((elem1 + elem2*elem3)**(1/2))

                # Compute metric
                metricSM = np.nansum(SMsMean * haversineDistance)
                
            else:
                metricSM = np.nan

            metricPerNeuron.append(metricSM)

        if 'CNN_UNET' in modelName:
            finalMetric = insertValues_CNN_UNET(metricPerNeuron)
        else:
            finalMetric = insertValues_CNNs(metricPerNeuron)

        saliencyMapMonth[idx, :, :] = finalMetric

    # Compute final mean
    SWSM_metric = np.nanmean(saliencyMapMonth, axis=0)

    # Save saliencyMaps as npy
    np.save(file = XAI_PATH + 'SM_SWSM_' + modelName + '_' + var + '_' + month + '_' + year + '.npy',
            arr = SWSM_metric)