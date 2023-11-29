#!/bin/bash

#SBATCH --job-name=call_INLA_PARD          #Your Job Name
#####SBATCH --partition=multinode
#SBATCH --partition=singlenode
#SBATCH --nodes=1                       #Number of Nodes desired e.g 1 node
#SBATCH --ntasks-per-core=1
####SBATCH --ntasks-per-node=1
####SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --time=19:59:00                 #Walltime: Duration for the Job to run HH:MM:SS
#SBATCH --exclusive
#SBATCH --error=%x.err          #The .error file name
#SBATCH --output=%x.out         #The .output file name
#####SBATCH --cpu-freq=2400000-2400000:performance         ### fix frequency


num_ranks=1

############################################################################


#ns=4002
ns=492
#nt=250
nt=50
#nb=2
nb=6
#no=7872
no=$((2*${ns}*${nt}))

solver_type=PARDISO
#solver_type=BTA

data_type=synthetic

############################################################################
TIME=$(date +%s)

export PARDISOLICMESSAGE=1
#export OMP_NESTED=TRUE
#export OMP_MAX_ACTIVE_LEVELS=2

l1t=0
l2t=32

# machine has 104 cores, so probably 8 x 8 = 64 current best setting. 
# significant increase in performance for pardiso until 16 threads, 32 only slightly faster
export OMP_NUM_THREADS=${l2t}
echo "OMP_NUM_THREADS=${l2t}"

#export OMP_DISPLAY_ENV=TRUE

#export MKL_NUM_THREADS=1
#echo "OMP_NUM_THREADS = ${omp_threads}"

export OMP_PROC_BIND=close
#export OMP_PLACES=cores

#export OMP_PROC_BIND=spread
#export OMP_PLACES=cores  #{0:36:1}

export OMP_PLACES={0:8:1},{18:8:1},{36:8:1},{54:8:1}
#export OMP_PLACES={0:16:1},{18:16:1},{36:16:1},{54:16:1}
##echo "OMP_PLACES={0:${l2t}:1}"

source ~/.profile

folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_nt${nt}_nb${nb}
#folder_path=/users/lgaedkem/b_INLA/data/${data_type}/ns${ns}_nt${nt}
# taskset -c 0-36

# CAREFUL : needs to be AT LEAST 11 (main + 10 workers, 10 because of hessian, for BFGS only 9 are required)
echo "srun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}" 
srun -n ${num_ranks}  ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >INLA_PARDISO_testPINNING_4NUMA_manualPinToCore_2pragmaLoop_singleNode_ns${ns}_nt${nt}_nb${nb}_${num_ranks}_${l1t}_${l2t}_${TIME}.txt 
#likwid-perfctr -C S0:0-15 -g MEM ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}

