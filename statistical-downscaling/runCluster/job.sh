#!/bin/bash

# Load some required modules (HPC cluster specific)
export PATH=$PATH:/bin/
source /etc/profile.d/modules.sh
module purge

udocker setup --nvidia --force $CONTAINER
nvidia-modprobe -u -c=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/

# Run job
udocker run --hostenv --hostauth --user=$USER \
-v $DIR_TO_MOUNT:/experiment/ $CONTAINER \
/bin/bash -c "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/;
              export _JAVA_OPTIONS="-Xmx10g";
              cd ./statistical-downscaling/;
              Rscript $R_SCRIPT"