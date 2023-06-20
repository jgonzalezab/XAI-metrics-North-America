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

# Auxiliar downscalePredict.keras
downscalePredict.keras.aux <- function(newdata,
                                       predElem,
                                       C4R.template,
                                       loss = 'mse') {
  
  x.global <- newdata$x.global
  n.mem <- length(x.global)
  pred <- lapply(1:n.mem, FUN = function(z) {
    predElem 
  })
  names(pred) <- paste("member", 1:n.mem, sep = "_")
  
  template <- C4R.template
  if (attr(newdata,"last.connection") == "dense") {
    ind <- attr(newdata,"indices_noNA_y")
    n.vars <- ncol(pred[[1]])/length(ind)
    if (isRegular(template)) {ncol.aux <- array3Dto2Dmat(template$Data) %>% ncol()} else {ncol.aux <- getShape(template,dimension = "loc")}
    pred <- lapply(1:n.mem,FUN = function(z) {
      aux <- matrix(nrow = nrow(pred[[z]]), ncol = ncol.aux)
      lapply(1:n.vars, FUN = function(zz) {
        aux[,ind] <- pred[[z]][,((ncol(pred[[1]])/n.vars)*(zz-1)+1):(ncol(pred[[1]])/n.vars*zz)]
        if (isRegular(template)) aux <- mat2Dto3Darray(aux, x = template$xyCoords$x, y = template$xyCoords$y)
        aux
      })
    })
  } 
  dimNames <- attr(template$Data,"dimensions")
  pred <- lapply(1:n.mem, FUN = function(z) {
    if (attr(newdata,"last.connection") == "dense") {
      lapply(1:n.vars, FUN = function(zz) {
        template$Data <- pred[[z]][[zz]]
        attr(template$Data,"dimensions") <- dimNames
        if (isRegular(template))  template <- redim(template, var = FALSE)
        if (!isRegular(template)) template <- redim(template, var = FALSE, loc = TRUE)
        return(template)
      }) %>% makeMultiGrid()
    } else {
      if (attr(newdata,"channels") == "first") n.vars <- dim(pred$member_1)[2]
      if (attr(newdata,"channels") == "last")  n.vars <- dim(pred$member_1)[4]
      lapply(1:n.vars, FUN = function(zz) {
        if (attr(newdata,"channels") == "first") template$Data <- pred[[z]] %>% aperm(c(2,1,3,4))
        if (attr(newdata,"channels") == "last")  template$Data <- pred[[z]] %>% aperm(c(4,1,2,3))
        template$Data <- template$Data[zz,,,,drop = FALSE]
        attr(template$Data,"dimensions") <- c("var","time","lat","lon")
        return(template)
      }) %>% makeMultiGrid()
    }
  })
  
  pred <- do.call("bindGrid",pred) %>% redim(drop = TRUE)
  pred$Dates <- attr(newdata,"dates")
  n.vars <- getShape(redim(pred,var = TRUE),"var")
  if (n.vars > 1) {
    if (loss == "gaussianLoss") {
      pred$Variable$varName <- c("mean","log_var")
    } else if (loss == "bernouilliGammaLoss") {
      pred$Variable$varName <- c("p","log_alpha","log_beta")
    } else {
      pred$Variable$varName <- paste0(pred$Variable$varName,1:n.vars)
    }
    pred$Dates <- rep(list(pred$Dates),n.vars)
    
  }
  return(pred)
}

# Load functions
source('./models.R')
source('./train.R')
source('./predict.R')

# PATHS
DATA_PATH <- '../../preprocessData/data/'
DATA_PATH_SD <- '../data/'
modelName <- 'LR'

# Load X and Y
load(paste0(DATA_PATH, 'x.rda'))
load(paste0(DATA_PATH, 'y.rda'))

# Split into train and test
yearsTrain <- 1980:2002
yearsTest <- 2003:2008

xTrain <- subsetGrid(x, years = yearsTrain)
yTrain <- subsetGrid(y, years = yearsTrain)

# Filter days with NaNs or NA
yTrain <- filterNA(yTrain)
xTrain <- intersectGrid(xTrain, yTrain, which.return = 1)

# Load land mask
yMask <- loadGridData(paste0(DATA_PATH, 'yMask.nc4'), var='tas')
yMask <- yMask$Data[1, , ]
yMask <- yMask * 0 + 1

# Apply land mask
yTrain <- applyMask(yTrain, yMask)

## Train set
print('Converting training predictions...')
# Load xyT and predictions
load(paste0(DATA_PATH_SD, 'xyT_DeepESD_train.rda'))

np <- import('numpy')
yPred <- np$load(paste0(DATA_PATH_SD, 'predLR_train.npy'))

# Format to nc4
predsTrain <- downscalePredict.keras.aux(newdata=xyT_train,
                                         predElem=yPred,
                                         C4R.template=yTrain)

# Save the prediction as netCDF
grid2nc(predsTrain, NetCDFOutFile = paste0(DATA_PATH_SD, 'predsTrain_',
                                           modelName, '.nc4'))

## Test set
print('Converting test predictions...')
# Load xyT and predictions
load(paste0(DATA_PATH_SD, 'xyT_DeepESD_test.rda'))

np <- import('numpy')
yPred <- np$load(paste0(DATA_PATH_SD, 'predLR_test.npy'))

# Format to nc4
predsTest <- downscalePredict.keras.aux(newdata=xyT_test,
                                        predElem=yPred,
                                        C4R.template=yTrain)

# Save the prediction as netCDF
grid2nc(predsTest, NetCDFOutFile = paste0(DATA_PATH_SD, 'predsTest_',
                                           modelName, '.nc4'))

## GCM Historical
print('Converting GCM Historical predictions...')
# Load xyT and predictions
load(paste0(DATA_PATH_SD, 'xyT_DeepESD_GCM_Hist.rda'))

np <- import('numpy')
yPred <- np$load(paste0(DATA_PATH_SD, 'predLR_GCM_Hist.npy'))

# Format to nc4
predsGCM_Hist <- downscalePredict.keras.aux(newdata=xyT_GCM_Hist,
                                            predElem=yPred,
                                            C4R.template=yTrain)

# Save the prediction as netCDF
grid2nc(predsGCM_Hist, NetCDFOutFile = paste0(DATA_PATH_SD, 'predsGCM_Hist_',
                                              modelName, '.nc4'))

## GCM Future
print('Converting GCM Future predictions...')
yearsPeriods <- c('2006_2040', '2041_2070', '2071_2100')

for (period in yearsPeriods) {

  # Load xyT and predictions
  load(paste0(DATA_PATH_SD, 'xyT_DeepESD_GCM_Fut_', period, '.rda'))

  np <- import('numpy')
  yPred <- np$load(paste0(DATA_PATH_SD, 'predLR_GCM_Fut_', period, '.npy'))

  # Format to nc4
  predsGCM_Fut <- downscalePredict.keras.aux(newdata=xyT_GCM_Fut,
                                             predElem=yPred,
                                             C4R.template=yTrain)

  # Save the prediction as netCDF
  grid2nc(predsGCM_Fut, NetCDFOutFile = paste0(DATA_PATH_SD, 'predsGCM_Fut_',
                                               period, '_',
                                               modelName, '.nc4'))

}