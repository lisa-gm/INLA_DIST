#!/bin/bash

#SBATCH --job-name=call_INLA_RGF           #Your Job Name
#SBATCH --nodes=3                      #Number of Nodes desired e.g 1 node
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

#ns=1186
#nt=500
#nb=5
#no=3230796

#ns=1105
ns=3240
nt=730
#nt=334
nb=6
#nb=8
#no=577778
#no=1931882
#no=13381997
#no=6540994
no=4735835

#ns=2307
#nt=201
#nb=6
#no=3885561

#ns=4795
#nt=50
#nb=5
#no=227117
#no=$((2*${ns}*${nt}))

#solver_type=$1
#solver_type=PARDISO
solver_type=RGF

#data_type=synthetic
data_type=temperature
	
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

folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_nt${nt}_nb${nb}
#folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_nt${nt}

source ~/.profile

echo "srun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}" 
srun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >INLA_RGF_${data_type}_ns${ns}_nt${nt}_nb${nb}_${num_ranks}_${l1t}_${l2t}_addDiag_relBFGSdelta_1e-7_pinned_i3.txt
#srun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >INLA_RGF_output_ns${ns}_nt${nt}_nb${nb}_${num_ranks}_${l1t}_${l2t}_singleCopyV.txt
#likwid-perfctr -C S0:0-15 -g MEM ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}

