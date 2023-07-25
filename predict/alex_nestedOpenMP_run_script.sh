#!/bin/bash

#SBATCH --job-name=call_INLA_RGF           #Your Job Name
#SBATCH --nodes=1                   #Number of Nodes desired e.g 1 nodea
#SBATCH --time=01:59:00                 #Walltime: Duration for the Job to run HH:MM:SS
#SBATCH --gres=gpu:a100:8
#SBATCH --partition=a100
###SBATCH --constraint=a100_80
###SBATCH --qos=a100multi
#SBATCH --exclusive
#SBATCH --error=%x.err          #The .error file name
#SBATCH --output=%x.out         #The .output file name

num_ranks=1

#ns=2252
#nt=0
#nb=2
#no=2252

#ns=567
#ns=1251
#nt_fit=14
#nt_pred=7
#nt_total=365
#no_per_ts=13567

ns=567
nss=567
#ns=1251
#nt_fit=16
nt_fit=30
nt_pred=7
nt_total=$((${nt_fit}+${nt_pred}))
#no_per_ts=$((2*${ns}))
no_per_ts=13567
#no=501979

#nt=30
#nb=2
nb=8
#no=1620
#no=1458
#no=$((2*${ns}*${nt}))
#no=2569965
#no=2472561

#solver_type=$1
#solver_type=PARDISO
solver_type=BTA

#data_type=synthetic
data_type=forecasting
	
export PARDISOLICMESSAGE=1
export OMP_NESTED=TRUE

# LAUNCH 10 MPI processes with x threads each. 8 or 16 threads for larger matrices seems appropriate.
# SEEMS 
# SLURM:
# --ntasks-per-node=
# -N : how nodes
# -n : how many processes per node
# --cpus-per-task=64 : how many threads per task
l1t=2
l2t=1

# machine has 104 cores, so probably 8 x 8 = 64 current best setting. 
# significant increase in performance for pardiso until 16 threads, 32 only slightly faster
export OMP_NUM_THREADS="${l1t},${l2t}"
echo "OMP_NUM_THREADS=${l1t},${l2t}"

#export OMP_NUM_THREADS="${l2t}"
#echo "OMP_NUM_THREADS=${l2t}"

#export MKL_NUM_THREADS=1
#echo "OMP_NUM_THREADS = ${omp_threads}"

#folder_path=~/b_INLA/data/forecasting/synthetic_wSpat/ns${ns}_ntFit${nt_fit}_ntPred5_noPerTs${no_per_ts}_nss${nss}_nb${nb}
#folder_path=~/b_INLA/data/forecasting/synthetic_wSpat/ns${ns}_ntFit${nt_fit}_ntPred${nt_pred}_noPerTs${no_per_ts}_nss${nss}_nb${nb}
folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_nss${nss}_ntFit${nt_fit}_ntPred${nt_pred}_nb${nb}
#folder_path=~/b_INLA/data/${data_type}/ns${ns}_nt${nt_total}_nb${nb}
#folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_nt${nt}_nb${nb}
#folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_nt${nt}

#source ~/.profile

# CAREFUL : needs to be AT LEAST 11 (main + 10 workers, 10 because of hessian, for BFGS only 9 are required)
echo "srun -n ${num_ranks} ./call_INLA ${ns} ${nss} ${nt_fit} ${nt_pred} ${nt_total} ${no} ${nb} ${folder_path} ${solver_type}" 
#srun -n ${num_ranks} ./call_INLA ${ns} ${nss} ${nt_fit} ${nt_pred} ${nt_total} ${no} ${nb} ${folder_path} ${solver_type} >output_ns${ns}_nss${nss}_ntFit${nt_fit}_ntPred${nt_pred}_nb${nb}_${solver_type}_cuda_testMSEyPred_${num_ranks}_${l1t}_${l2t}.txt #>INLA_RGF_output_ns${ns}_nt${nt}_nb${nb}_${num_ranks}_${l1t}_${l2t}_newRGFClass_nestedOMP_singleCopyV_i1.txt
srun -n ${num_ranks} ./call_INLA ${ns} ${nss} ${nt_fit} ${nt_pred} ${nt_total} ${no_per_ts} ${nb} ${folder_path} ${solver_type} >output_ns${ns}_nss${nss}_ntFit${nt_fit}_ntPred${nt_pred}_nb${nb}_${solver_type}_cuda_${num_ranks}_${l1t}_${l2t}.txt #>INLA_RGF_output_ns${ns}_nt${nt}_nb${nb}_${num_ranks}_${l1t}_${l2t}_newRGFClass_nestedOMP_singleCopyV_i1.txt
#mpirun -n ${num_ranks} nsys profile -o nsys_output_checkBTAinv ./call_INLA ${ns} ${nb} ${nt_fit} ${nt_pred} ${nt_total} ${no_per_ts} ${folder_path} ${solver_type} #>output_forecasting_ns${ns}_nt${nt}_nb${nb}_wNA.txt #>INLA_RGF_output_ns${ns}_nt${nt}_nb${nb}_${num_ranks}_${l1t}_${l2t}_newRGFClass_nestedOMP_singleCopyV_i1.txt
#srun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >INLA_RGF_output_ns${ns}_nt${nt}_nb${nb}_${num_ranks}_${l1t}_${l2t}_singleCopyV.txt
#likwid-perfctr -C S0:0-15 -g MEM ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}

