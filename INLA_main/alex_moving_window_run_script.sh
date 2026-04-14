#!/bin/bash

#SBATCH --job-name=call_INLA_moving_window 
#SBATCH --nodes=1                  #Number of Nodes desired e.g 1 nodea
#SBATCH --time=01:59:00                 #Walltime: Duration for the Job to run HH:MM:SS
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
###SBATCH --constraint=a100_80
####SBATCH --qos=a100multi
####BATCH --exclusive
#SBATCH --error=%x.err          #The .error file name
#SBATCH --output=%x.out         #The .output file name

num_ranks=1

BASE_PATH=$HOME

ns=3046
nss=0
ntFit=153  ### total number of days in the full dataset
nt_window=30
ntPred=0
# nt=$((${ntFit}+${ntPred}))
nb=2
#no=30240
#noPerTs=1008 #
year=2022

#########################
# 3 SETUPS: validation patch removal, validation random removal, full dataset with moving window
# uncomment as needed
######## SETUP 1: validation patch removal
#val_tag=val_patch_
#val=1
#noPerTs=29130    ## total number of observations

######## SETUP 2: validation random removal
#val_tag=val_random_
#val=1
#noPerTs=24192  ## total number of observations

######## SETUP 3: full dataset with moving window
val_tag= # make it empty for no validation
val=0
noPerTs=1008  ### will get multiplied by number of days internally
#########################

solver_type=BTA
data_type=swiss_rainfall
likelihood=binomial

export OMP_NESTED=TRUE

l1t=1
l2t=16

export OMP_NUM_THREADS="${l1t},${l2t}"
echo "OMP_NUM_THREADS=${l1t},${l2t}"

# Generate timestamp for output file
timestamp=$(date +%Y%m%d_%H%M%S)

### TODO: change the path to the data folder as needed
folder_path=${BASE_PATH}/data/${data_type}/${likelihood}/${val_tag}year${year}_ns${ns}_nt${nt_window}_nb${nb}
#folder_path=${BASE_PATH}/repositories/approx_non_stationary_models/rainfall_dataset/data/year${year}_ns${ns}_nt${ntFit}_nb${nb} #_year${year}


########## RUN THE CODE
echo "srun -n ${num_ranks} ./call_INLA_moving_window ${ns} ${ntFit} ${nss} ${nb} ${val} ${noPerTs} ${likelihood} ${folder_path} ${solver_type}" 
srun -n ${num_ranks} ./call_INLA_moving_window ${ns} ${ntFit} ${nss} ${nb} ${val} ${noPerTs} ${likelihood} ${folder_path} ${solver_type} >INLA_${solver_type}_output_moving_window_mixedStrategy_ns${ns}_ntWindow${nt_window}_ntPred0_nss${nss}_nb${nb}_${likelihood}_${num_ranks}_${l1t}_${l2t}_${timestamp}.txt #${timestamp}

