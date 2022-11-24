### Train the models and compute predictions
We provide the code to train the different deep learning models and compute the corresponding predictions for both reanalysis and GCM data. The [computeModel.R](https://github.com/jgonzalezab/XAI-metrics-North-America/blob/main/statistical-downscaling/computeModel.R) script calls the functions to perform these operations. By tuning some of its variables we control what model to run. For example the DeepESD model requires the following values:

```
# Train and save a model on the data
modelName <- 'DeepESD'
connections <- c('conv', 'dense')
```

Whereas CNN-UNET:
```
# Train and save a model on the data
modelName <- 'CNN_UNET'
connections <- c('conv', 'conv')
```

For all these operations we rely on [climate4R](https://github.com/SantanderMetGroup/climate4R). As for the downloading and preprocessing of data, we also provide the [runCluster](https://github.com/jgonzalezab/XAI-metrics-North-America/tree/main/statistical-downscaling/runCluster) folder to ease the training of these models in GPUs within HPC clusters.
