# Mask observations (land gridpoints)
applyMask <- function(grid, mask) {

  for (i in 1:dim(grid$Data)[1]) {
    grid$Data[i, , ] <- grid$Data[i, , ] * mask
  }

  return(grid)

}

# Train and save a model
trainCNN <- function(xTrain, yTrain, yMask,
                     modelName,
                     connections) {
    
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

    # Prepare data for the model
    xyT <- prepareData.keras(x = xTrain_stand,
                             y = yTrain,
                             first.connection = connections[1],
                             last.connection = connections[2],
                             channels = 'last')

    # Load the model to train
    model <- load_model(model = modelName,
                        input_shape = dim(xyT$x.global)[-1],
                        output_shape = dim(xyT$y$Data)[2])

    print('****************************************')
    print(paste0('Training model ', modelName))
    print(paste0('Number of parameters: ', count_params(model)))
    print('****************************************')

    # Train and save the best model
    modelPathName <- paste0(MODELS_PATH, modelName, '.h5')
    
    downscaleTrain.keras(obj = xyT,
                         model = model,
                         clear.session = TRUE,
                         compile.args = list('loss' = 'mse',
                                             'optimizer' = optimizer_adam(lr = 0.0001)),
                         fit.args = list('batch_size' = 100,
                                         'epochs' = 10000, 
                                         'validation_split' = 0.1,
                                         'verbose' = 1,
                                         'callbacks' = list(callback_early_stopping(patience = 30), 
                                                            callback_model_checkpoint(
                                                            filepath = modelPathName,
                                                            monitor = 'val_loss', save_best_only = TRUE,
                                                            save_weights_only = TRUE))))

    # Save xyT object to compute predictions
    save(xyT, file = paste0(DATA_PATH_SD, 'xyT_', modelName, '.rda'))

}
