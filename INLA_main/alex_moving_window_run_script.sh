#!/bin/bash

#SBATCH --job-name=call_INLA_moving_window # Pardiso #          #Your Job Name
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


# ns=4002
# nss=0
# ntFit=30
# ntPred=0
# nt=$((${ntFit}+${ntPred}))
# nb=6
# no=$((2*${ns}*${ntFit}))

#nss=642
#nt=30
#nb=2
#nb=6
#no=7872
#no=$((2*${ns}*${ntFit}))

ns=3046
nss=0
ntFit=214
nt_window=30
ntPred=0
nt=$((${ntFit}+${ntPred}))
nb=2
#no=30240
noPerTs=30195
year=2022

#solver_type=$1
#solver_type=PARDISO
solver_type=BTA
#solver_type=Eigen

#data_type=swiss_rainfall
#data_type=regression
data_type=synthetic

likelihood=binomial
#likelihood=gaussian

export OMP_NESTED=TRUE

l1t=1
l2t=16

export OMP_NUM_THREADS="${l1t},${l2t}"
echo "OMP_NUM_THREADS=${l1t},${l2t}"

# Generate timestamp for output file
timestamp=$(date +%Y%m%d_%H%M%S)

folder_path=/home/hpc/ihpc/ihpc060h/repositories/approx_non_stationary_models/rainfall_dataset/data/cross_val_year${year}_ns${ns}_nt${ntFit}_nb${nb} #_year${year}


echo "srun -n ${num_ranks} ./call_INLA_moving_window ${ns} ${ntFit} ${nss} ${nb} ${noPerTs} ${likelihood} ${folder_path} ${solver_type}" 
srun -n ${num_ranks} ./call_INLA_moving_window ${ns} ${ntFit} ${nss} ${nb} ${noPerTs} ${likelihood} ${folder_path} ${solver_type} >INLA_${solver_type}_output_moving_window_ns${ns}_ntWindow${nt_window}_ntPred0_nss${nss}_nb${nb}_${likelihood}_${num_ranks}_${l1t}_${l2t}_${timestamp}.txt #${timestamp}

