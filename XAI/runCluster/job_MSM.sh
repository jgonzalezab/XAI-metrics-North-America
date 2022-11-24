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

# Coordinates from which saliency maps are computed
DIMROW=('15' '65' '110')
DIMCOL=('121' '140' '35')

# Run job
for model in "${MODELS[@]}"
do
  echo $model
  for i in {0..2}
  do
      echo ${DIMROW[$i]}
      echo ${DIMCOL[$i]}
      udocker run --hostenv --hostauth --user=$USER \
      -v $DIR_TO_MOUNT:/experiment/ $CONTAINER \
      /bin/bash -c "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/;
                    cd ./XAI/;
                    python compute_MSM.py ${model} ${DIMROW[$i]} ${DIMCOL[$i]}"
   done
done