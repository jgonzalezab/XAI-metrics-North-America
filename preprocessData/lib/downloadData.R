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

# Download reanalysis data
download_X <- function(dataset, vars, lon, lat, years) {

  x <- lapply(vars, function(var) {
     print(paste0('Downloading variable ', var, ' of ', dataset, ' dataset'))
 	   loadGridData(dataset = dataset,
 	                var = var,
 	                lonLim = lon,
 	                latLim = lat,
 	                years = years)
 	 }) %>% makeMultiGrid()

  print(paste0('Saving as rda in ', DATA_PATH))
  save(x, file = paste0(DATA_PATH, 'x.rda'))

}

# Download observation data
download_Y <- function(dataset, lon, lat, years) {

  y <- loadGridData(dataset = dataset,
 	                var = 'tas',
 	                lonLim = lon,
 	                latLim = lat,
 	                years = years)

  print(paste0('Saving as rda in ', DATA_PATH))
  save(y, file = paste0(DATA_PATH, 'y.rda'))

  print(paste0('Saving as netCDF in ', DATA_PATH))
  grid2nc(y, NetCDFOutFile=paste0(DATA_PATH, 'y.nc4'))

}

# EC-EARTH predictors in the historical period
download_GCM_predictors_historical <- function(dataset, xRef, vars, lon, lat, years) {

  x_GCM <- lapply(vars, function(var) {
         print(paste0('Downloading variable ', var, ' of ', dataset, ' dataset'))
 	   		 loadGridData(dataset = dataset,
 	                	 	var = var,
 	                	    lonLim = lon,
 	                	 	latLim = lat,
 	                	 	years = years) %>%
 	   						interpGrid(new.coordinates = getGrid(xRef), method = 'bilinear')
 	 			 }) %>% makeMultiGrid()

  print(paste0('Saving as rds in ', DATA_PATH))
  saveRDS(x_GCM, file = paste0(DATA_PATH, 'x_GCM_historical.rds'))

}

# EC-EARTH predictors in future periods
download_GCM_predictors_future <- function(dataset, xRef, vars, lon, lat, years) {

  x_GCM <- lapply(vars, function(var) {
         print(paste0('Downloading variable ', var, ' of ', dataset, ' dataset'))
 	   		 loadGridData(dataset = dataset,
 	                	 	var = var,
 	                	    lonLim = lon,
 	                	 	latLim = lat,
 	                	 	years = years) %>%
 	   						interpGrid(new.coordinates = getGrid(xRef), method = 'bilinear')
 	 			 }) %>% makeMultiGrid()

  print(paste0('Saving as rds in ', DATA_PATH))
  saveRDS(x_GCM, file = paste0(DATA_PATH, 'x_GCM_future_', years[1], '_', years[length(years)], '.rds'))

}

# EC-EARTH proyections in the historical period
download_GCM_proyections_historical <- function(dataset, yRef, lon, lat, years) {

  y_GCM <- loadGridData(dataset = dataset,
 	                	var = 'tas',
 	                	lonLim = lon,
 	                	latLim = lat,
 	                	years = years) %>%
 	   					interpGrid(new.coordinates = getGrid(yRef), method = 'bilinear')

  print(paste0('Saving as netCDF in ', DATA_PATH))
  grid2nc(y_GCM, NetCDFOutFile=paste0(DATA_PATH, 'y_GCM_historical.nc4'))

}

# EC-EARTH proyections in future periods
download_GCM_proyections_future <- function(dataset, yRef, lon, lat, years) {

  y_GCM <- loadGridData(dataset = dataset,
 	                	var = 'tas',
 	                	lonLim = lon,
 	                	latLim = lat,
 	                	years = years) %>%
 	   					interpGrid(new.coordinates = getGrid(yRef), method = 'bilinear')

  print(paste0('Saving as netCDF in ', DATA_PATH))
  grid2nc(y_GCM,
  		  NetCDFOutFile=paste0(DATA_PATH, 'y_GCM_future_', years[1], '_', years[length(years)], '.nc4'))

}
