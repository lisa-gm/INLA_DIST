#!/bin/bash

#SBATCH --job-name=call_INLA_RGF           #Your Job Name
#SBATCH --nodes=3                       #Number of Nodes desired e.g 1 node
#SBATCH --time=19:59:00                 #Walltime: Duration for the Job to run HH:MM:SS
#SBATCH --gres=gpu:a100:8
#SBATCH --partition=a100
#SBATCH --qos=a100multi
#SBATCH --exclusive
#SBATCH --error=%x.err          #The .error file name
#SBATCH --output=%x.out         #The .output file name

num_ranks=9

#ns=2252
#nt=0
#nb=2
#no=2252

#ns=$1
#nt=$2
#nb=$3
#no=$4

#ns=1002
#nt=16
#nb=2
#no=188242

#ns=19362
ns=4002
nt=250
#nt=30
#nb=2
nb=6
#no=7872
no=$((2*${ns}*${nt}))

#solver_type=$1
#solver_type=PARDISO
solver_type=RGF

data_type=synthetic

	
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
l2t=4

# machine has 104 cores, so probably 8 x 8 = 64 current best setting. 
# significant increase in performance for pardiso until 16 threads, 32 only slightly faster
export OMP_NUM_THREADS="${l1t},${l2t}"
echo "OMP_NUM_THREADS=${l1t},${l2t}"

#export OMP_NUM_THREADS="${l2t}"
#echo "OMP_NUM_THREADS=${l2t}"

#export MKL_NUM_THREADS=1
#echo "OMP_NUM_THREADS = ${omp_threads}"

folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_nt${nt}_nb${nb}
#folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_nt${nt}

source ~/.profile

# CAREFUL : needs to be AT LEAST 11 (main + 10 workers, 10 because of hessian, for BFGS only 9 are required)
echo "srun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}" 
srun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >INLA_RGF_output_ns${ns}_nt${nt}_nb${nb}_${num_ranks}_${l1t}_${l2t}_newRGFClass_nestedOMP_singleCopyV_i1.txt
#srun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >INLA_RGF_output_ns${ns}_nt${nt}_nb${nb}_${num_ranks}_${l1t}_${l2t}_singleCopyV.txt
#likwid-perfctr -C S0:0-15 -g MEM ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}
