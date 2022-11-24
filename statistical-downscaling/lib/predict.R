# Scaling delta mapping transformation
scalingDeltaMapping <- function(grid, base, ref) {

    # Remove the seasonal trend
    grid_detrended <- scaleGrid(grid,
                                base = grid,
                                ref = base,
                                type = "center",
                                spatial.frame = "gridbox",
                                time.frame = "monthly")

    # Bias correct the mean and variance
    grid_detrended_corrected <- scaleGrid(grid_detrended,
                                          base = base,
                                          ref = ref,
                                          type = "standardize",
                                          spatial.frame = "gridbox",
                                          time.frame = "monthly")
    
    # Add the seasonal trend
    grid_corrected <- scaleGrid(grid_detrended_corrected,
                                base = base,
                                ref = grid,
                                type = "center",
                                spatial.frame = "gridbox",
                                time.frame = "monthly")

    return(grid_corrected)
}

# Compute predictions on the training set (reanalysis predictors)
predictTrain <- function(xTrain, yTrain, modelName, region=NULL) {

    # Filter days with NaNs or NA
    yTrain <- filterNA(yTrain)
    xTrain <- intersectGrid(xTrain, yTrain, which.return = 1)

    # Apply land mask or substitute NANs by zero if training CNN_UNET
    if (modelName == 'CNN_UNET') {
        yTrain$Data[is.na(yTrain$Data)] <- 0
    } else {
        yTrain <- applyMask(yTrain, yMask)
    }

    # Standardize predictors
    xTrain_stand <- scaleGrid(xTrain, type = 'standardize')

    # Load xyT
    load(file = paste0(DATA_PATH_SD, 'xyT_', modelName, '.rda'))

    # Prepare xyT
	xyT_train <- prepareNewData.keras(xTrain_stand, xyT)

    # Load model
    modelPathName <- paste0(MODELS_PATH, modelName, '.h5')

    model <- load_model(model = modelName,
                        input_shape = dim(xyT$x.global)[-1],
                        output_shape = dim(xyT$y$Data)[2])

    model <- load_model_weights_hdf5(object = model,
									 filepath = modelPathName)

    # Compute predictions
    predsTrain <- downscalePredict.keras(xyT_train,
                               	 	     model = model,
                               	 	     loss = 'mse',
                               	 	     C4R.template = yTrain,
                               	 	     clear.session = TRUE)

    # Save the prediction as netCDF
    grid2nc(predsTrain, NetCDFOutFile = paste0(DATA_PATH_SD, 'predsTrain_',
                                               modelName, '.nc4'))

}

# Compute predictions on the test set (reanalysis predictors)
predictTest <- function(xTrain, yTrain, xTest, yTest, modelName) {

    # Filter days with NaNs or NA
    yTrain <- filterNA(yTrain)
    xTrain <- intersectGrid(xTrain, yTrain, which.return = 1)

    # Apply land mask or substitute NANs by zero if training CNN_UNET
    if (modelName == 'CNN_UNET') {
        yTrain$Data[is.na(yTrain$Data)] <- 0
    } else {
        yTrain <- applyMask(yTrain, yMask)
    }

    # Standardize predictors
    xTest_stand <- scaleGrid(xTest, xTrain, type = 'standardize')

    # Load xyT
    load(file = paste0(DATA_PATH_SD, 'xyT_', modelName, '.rda'))

    # Prepare xyT
	xyT_test <- prepareNewData.keras(xTest_stand, xyT)

    # Load model
    modelPathName <- paste0(MODELS_PATH, modelName, '.h5')

    model <- load_model(model = modelName,
                        input_shape = dim(xyT$x.global)[-1],
                        output_shape = dim(xyT$y$Data)[2])
                        
    model <- load_model_weights_hdf5(object = model,
									 filepath = modelPathName)

    # Compute predictions
    predsTest <- downscalePredict.keras(xyT_test,
                               	 	    model = model,
                               	 	    loss = 'mse',
                               	 	    C4R.template = yTrain,
                               	 	    clear.session = TRUE)

    # Save the prediction as netCDF
    grid2nc(predsTest, NetCDFOutFile = paste0(DATA_PATH_SD, 'predsTest_',
                                              modelName, '.nc4'))
	
}

# Compute predictions on the GCM predictors over the historical period
predictGCM_Hist <- function(xTrain, yTrain, modelName) {

    # Filter days with NaNs or NA
    yTrain <- filterNA(yTrain)
    xTrain <- intersectGrid(xTrain, yTrain, which.return = 1)

    # Apply land mask or substitute NANs by zero if training CNN_UNET
    if (modelName == 'CNN_UNET') {
        yTrain$Data[is.na(yTrain$Data)] <- 0
    } else {
        yTrain <- applyMask(yTrain, yMask)
    }

    # Load xyT
    load(file = paste0(DATA_PATH_SD, 'xyT_', modelName, '.rda'))

    # Load GCM historical data
    x_GCM_Hist <- readRDS(paste0(DATA_PATH, 'x_GCM_historical.rds'))

    # Apply transformation to GCM predictors
    x_GCM_Hist_BC <- scalingDeltaMapping(grid=x_GCM_Hist, base=x_GCM_Hist, ref=xTrain)

    # Standardize
    x_GCM_Hist_BC_stand <- scaleGrid(x_GCM_Hist_BC, xTrain, type='standardize')

    # Prepare xyT
	xyT_GCM_Hist <- prepareNewData.keras(x_GCM_Hist_BC_stand, xyT)

    # Load model
    modelPathName <- paste0(MODELS_PATH, modelName, '.h5')

    model <- load_model(model = modelName,
                        input_shape = dim(xyT$x.global)[-1],
                        output_shape = dim(xyT$y$Data)[2])
                        
    model <- load_model_weights_hdf5(object = model,
									 filepath = modelPathName)

    # Compute predictions
    predsGCM_Hist <- downscalePredict.keras(xyT_GCM_Hist,
                               	 	        model = model,
                               	 	        loss = 'mse',
                               	 	        C4R.template = yTrain,
                               	 	        clear.session = TRUE)

    # Save the prediction as netCDF
    grid2nc(predsGCM_Hist, NetCDFOutFile = paste0(DATA_PATH_SD, 'predsGCM_Hist_',
                                                  modelName, '.nc4'))

}

# Compute predictions on the GCM predictors over all the future periods
predictGCM_Fut <- function(xTrain, yTrain, modelName) {

    # Filter days with NaNs or NA
    yTrain <- filterNA(yTrain)
    xTrain <- intersectGrid(xTrain, yTrain, which.return = 1)

    # Apply land mask or substitute NANs by zero if training CNN_UNET
    if (modelName == 'CNN_UNET') {
        yTrain$Data[is.na(yTrain$Data)] <- 0
    } else {
        yTrain <- applyMask(yTrain, yMask)
    }

    # Load xyT
    load(file = paste0(DATA_PATH_SD, 'xyT_', modelName, '.rda'))

    # Load GCM historical data
    x_GCM_Hist <- readRDS(paste0(DATA_PATH, 'x_GCM_historical.rds'))

    # Load model
    modelPathName <- paste0(MODELS_PATH, modelName, '.h5')

    model <- load_model(model = modelName,
                        input_shape = dim(xyT$x.global)[-1],
                        output_shape = dim(xyT$y$Data)[2])
                        
    model <- load_model_weights_hdf5(object = model,
									 filepath = modelPathName)

    # Periods
    yearsPeriods <- c('2006_2040', '2041_2070', '2071_2100')

    # Iterate over these periods
    for (period in yearsPeriods) {
        
        print(paste0('Computing GCM prediction on ', period))

        # Load GCM future data
        x_GCM_Fut <- readRDS(paste0(DATA_PATH, 'x_GCM_future_', period, '.rds'))

        # Apply transformation to GCM predictors
        x_GCM_Fut_BC <- scalingDeltaMapping(grid=x_GCM_Fut, base=x_GCM_Hist, ref=xTrain)

        # Standardize
        x_GCM_Fut_BC_stand <- scaleGrid(x_GCM_Fut_BC, xTrain, type='standardize')

        # Prepare xyT
        xyT_GCM_Fut <- prepareNewData.keras(x_GCM_Fut_BC_stand, xyT)

        # Compute predictions
        predsGCM_Fut <- downscalePredict.keras(xyT_GCM_Fut,
                                               model = model,
                                               loss = 'mse',
                                               C4R.template = yTrain,
                                               clear.session = TRUE)

        # Save the prediction as netCDF
        grid2nc(predsGCM_Fut, NetCDFOutFile = paste0(DATA_PATH_SD, 'predsGCM_Fut_',
                                                     period, '_', modelName, '.nc4'))

    }

}