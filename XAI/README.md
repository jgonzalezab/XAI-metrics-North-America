### Compute saliency maps
Thhis folder contains the code to compute the XAI metrics for the different models. For this part of the paper we use Python instead of R, since [INNvestigate](https://github.com/albermax/innvestigate) is only available for the former. Due to incompabilities issues between versions of different libraries, this environment must be reproduced with its own Dockerfile ([Dockerfile_XAI](https://github.com/jgonzalezab/XAI-metrics-North-America/blob/main/docker/Dockerfile_XAI)).

Similar to `computeModel.R`, scripts [compute_MSM.py](https://github.com/jgonzalezab/XAI-metrics-North-America/blob/main/XAI/compute_MSM.py) and [compute_ISM_SWSM.py](https://github.com/jgonzalezab/XAI-metrics-North-America/blob/main/XAI/compute_ISM_SWSM.py) control the computation of the mean saliency map and ISM and SWSM metrics. For example, to compute ISM and SWSM for DeepESD at August of the year 2000, we must run the following:

```
python compute_ISM_SWSM.py DeeepESD 08 2000
```

It saves the results as `.npy` in the [XAI/data/](https://github.com/jgonzalezab/XAI-metrics-North-America/tree/main/XAI/data) folder. As in previous parts, we provide the [runCluster](https://github.com/jgonzalezab/XAI-Statistical-Downscaling/tree/main/XAI/runCluster) folder to ease the computation on HPC clusters.
