### Download and preprocess data

All the code for downloading and preprocessing the data is available here. Note that downloading the data may require an account at [UDG-TAP](http://meteo.unican.es/udg-tap/home), however data can also be downloaded from the URLs of *Section 2.1 Region of study and data* of the paper.

All data can be downloaded by running [getData.R](https://github.com/jgonzalezab/XAI-metrics-North-America/blob/main/preprocessData/getData.R). This task is time-consuming and requires about 16GB of disk space. While the data is being downloaded, the interpolation required by some of these datasets is also applied.

In the folder [runCluster](https://github.com/jgonzalezab/XAI-Statistical-Downscaling/tree/main/preprocessData/runCluster) we provide two `.sh` scripts to run this part of the paper following the workflow developed in [*A Container-Based Workflow for Distributed Training of Deep Learning Algorithms in HPC Clusters*](https://doi.org/10.1007/s10586-022-03798-7). This workflow help us to run experiments in HPC clusters using Docker when GPUs are involved. For all these operations we rely on [climate4R](https://github.com/SantanderMetGroup/climate4R).
