#!/bin/bash

#SBATCH --job-name=call_INLA_mpi           #Your Job Name
#SBATCH --nodes=3                       #Number of Nodes desired e.g 1 node
#SBATCH --ntasks-per-node=1              #MPI task per node
#SBATCH --cpus-per-task=4               # 16 cores per process
#SBATCH --time=00:05:00                 #Walltime: Duration for the Job to run HH:MM:SS
#SBATCH --gres=gpu:1
#SBATCH --error=%x.err          #The .error file name
#SBATCH --output=%x.out         #The .output file name

num_ranks=3

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

#ns=1002
ns=492
#nt=0
nt=16
#nt=25
nb=2
#no=25050
no=7872
#no=12873

#solver_type=$1
solver_type=PARDISO
#solver_type=RGF

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
l2t=1

# machine has 104 cores, so probably 8 x 8 = 64 current best setting. 
# significant increase in performance for pardiso until 16 threads, 32 only slightly faster
export OMP_NUM_THREADS="${l1t},${l2t}"
echo "OMP_NUM_THREADS=${l1t},${l2t}"

#export MKL_NUM_THREADS=1
#echo "OMP_NUM_THREADS = ${omp_threads}"

#folder_path=/home/x_gaedkelb/b_INLA/data/temperature/ns${ns}_nt${nt}
folder_path=/home/x_gaedkelb/b_INLA/data/${data_type}/ns${ns}_nt${nt}

# CAREFUL : needs to be AT LEAST 11 (main + 10 workers, 10 because of hessian, for BFGS only 9 are required)
echo "mpirun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}" 
mpirun -np ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >${folder_path}/INLA_PARDISO_output.txt
#likwid-perfctr -C S0:0-15 -g MEM ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}

