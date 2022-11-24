#! /bin/bash

# Container name
export CONTAINER="saliency-maps"

# Job config
export JOB_NAME=""
export JOB2RUN="./job_ISM_SWSM.sh"
export DIR_TO_MOUNT="/XAI-metrics-North-America"

# Create slurm-log folder if it does not already exist
mkdir -p "${DIR_TO_MOUNT}/XAI/cluster-log/"

# SLURM parameters
SLURM_PARAMS="--job-name=${JOB_NAME} --partition=wngpu --gres=gpu:1
              --time=72:00:00 --mem-per-cpu=64000 --output=${DIR_TO_MOUNT}/XAI/cluster-log/${JOB_NAME}.out
              --error=${DIR_TO_MOUNT}/XAI/cluster-log/${JOB_NAME}.err"
              
sbatch $SLURM_PARAMS $JOB2RUN