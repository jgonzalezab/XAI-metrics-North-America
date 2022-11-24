#!/bin/bash

# Load some required modules (HPC cluster specific)
export PATH=$PATH:/bin/
source /etc/profile.d/modules.sh
module purge

udocker setup --nvidia --force $CONTAINER
nvidia-modprobe -u -c=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/

# Models to compute saliency maps from
MODELS=('DeepESD')

# Dates to iterate over
MONTH=('08' '12')
YEAR=('2000' '2000')

# Run job
for model in "${MODELS[@]}"
do
  echo $model
  for i in {0..1}
  do
      echo ${MONTH[$i]}
      echo ${YEAR[$i]}
      udocker run --hostenv --hostauth --user=$USER \
      -v $DIR_TO_MOUNT:/experiment/ $CONTAINER \
      /bin/bash -c "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/;
                    cd ./XAI/;
                    python compute_ISM_SWSM.py ${model} ${MONTH[$i]} ${YEAR[$i]}"
   done
done