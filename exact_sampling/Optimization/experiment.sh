#!/bin/bash
#PBS -lselect=1:ncpus=1:mem=96gb
#PBS -lwalltime=12:00:00
#PBS -J 1-10

export OPT_TYPES="FD_GD CFD_GD Our_GD"
export SIG="1"
export STEP_SIZE="1e-3"
export h="1"

export PATH=$HOME/miniconda3/bin/:$PATH
export PATH_TO_ADV_FOLDER=$HOME/new_adventure
source activate
conda activate simplex_grad

# CUTEst
export ARCHDEFS="/rds/general/user/dl2119/home/cutest/archdefs/"
export SIFDECODE="/rds/general/user/dl2119/home/cutest/sifdecode"
export CUTEST="/rds/general/user/dl2119/home/cutest/cutest"
export MASTSIF="/rds/general/user/dl2119/home/cutest/mastsif/"
export PATH="${SIFDECODE}/bin:${PATH}"
export PATH="${CUTEST}/bin:${PATH}"
export MANPATH="${SIFDECODE}/man:${MANPATH}"
export MANPATH="${CUTEST}/man:${MANPATH}"
export MYARCH="pc.lnx.gfo"

export PYCUTEST_CACHE="${HOME}/CUTEst/pycutest_cache"


python $HOME/curr_adventure/exact_sampling/Optimization/QuadraticOptimization.py