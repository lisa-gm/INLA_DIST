#!/bin/bash

#SBATCH --job-name=call_INLA_BTA # Pardiso #          #Your Job Name
#SBATCH --nodes=1                  #Number of Nodes desired e.g 1 nodea
#SBATCH --time=00:59:00                 #Walltime: Duration for the Job to run HH:MM:SS
#SBATCH --gres=gpu:a100:8
#SBATCH --partition=a100
###SBATCH --constraint=a100_80
####SBATCH --qos=a100multi
#SBATCH --exclusive
#SBATCH --error=%x.err          #The .error file name
#SBATCH --output=%x.out         #The .output file name

num_ranks=7

#ns=0
#nss=0
#ntFit=0
#ntPred=0
#nt=$((${ntFit}+${ntPred}))
#nb=6
#no=200

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

ns=3044
nss=0
ntFit=30
ntPred=0
nt=$((${ntFit}+${ntPred}))
nb=2
no=30240

#solver_type=$1
#solver_type=PARDISO
solver_type=BTA
#solver_type=Eigen

#data_type=swiss_rainfall
#data_type=regression
data_type=synthetic

likelihood=binomial
#likelihood=gaussian

export PARDISOLICMESSAGE=1
export OMP_NESTED=TRUE

l1t=1
l2t=16

# machine has 104 cores, so probably 8 x 8 = 64 current best setting. 
# significant increase in performance for pardiso until 16 threads, 32 only slightly faster
export OMP_NUM_THREADS="${l1t},${l2t}"
echo "OMP_NUM_THREADS=${l1t},${l2t}"

#export OMP_NUM_THREADS="${l2t}"
#echo "OMP_NUM_THREADS=${l2t}"

# Generate timestamp for output file
timestamp=$(date +%Y%m%d_%H%M%S)

folder_path=/home/hpc/ihpc/ihpc060h/repositories/approx_non_stationary_models/rainfall_dataset/data/ns${ns}_nt${nt}_nb${nb}

#folder_path=/home/hpc/ihpc/ihpc060h/data/${data_type}/${likelihood}/ns${ns}_nt${nt}_nb${nb}
#folder_path=/home/hpc/ihpc/ihpc060h/data/${data_type}/ns${ns}_nt${nt}_nb${nb}


echo "srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type}" 
srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type} >INLA_${solver_type}_output_ns${ns}_ntFit${ntFit}_ntPred0_nss${nss}_nb${nb}_${likelihood}_${num_ranks}_${l1t}_${l2t}_${timestamp}.txt
#srun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >INLA_RGF_output_ns${ns}_nt${nt}_nb${nb}_${num_ranks}_${l1t}_${l2t}_singleCopyV.txt
#likwid-perfctr -C S0:0-15 -g MEM ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}

