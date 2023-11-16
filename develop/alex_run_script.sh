#!/bin/bash

#SBATCH --job-name=call_INLA_RGF           #Your Job Name
#SBATCH --nodes=1                   #Number of Nodes desired e.g 1 nodea
#SBATCH --time=00:59:00                 #Walltime: Duration for the Job to run HH:MM:SS
#SBATCH --gres=gpu:a100:8
#SBATCH --partition=a100
###SBATCH --constraint=a100_80
###SBATCH --qos=a100multi
#SBATCH --exclusive
#SBATCH --error=%x.err          #The .error file name
#SBATCH --output=%x.out         #The .output file name

num_ranks=1

#ns=0
#nss=0
#ntFit=0
#ntPred=0
#nt=$((${ntFit}+${ntPred}))
#nb=6
#no=200

#ns=492
#nss=0
#ntFit=50
#ntPred=0
#nt=$((${ntFit}+${ntPred}))
#nb=6
#noPerTs=$((2*${ns}))
#no=$((2*${ns}*${ntFit}))


ns=425
#nss=1442
#ns=20252
#ns=16002
#ntFit=60
nss=0
#ns=642
ntFit=0
ntPred=0
nt=$((${ntFit}+${ntPred}))
#nss=642
#nt=30
#nb=2
nb=3
#no=7872
#no=$((2*${ns}*${ntFit}))
#noPerTs=$((2*${ns}))
no=595

#solver_type=$1
#solver_type=PARDISO
solver_type=BTA
#solver_type=Eigen

data_type=synthetic
#data_type=regression
	
likelihood=gaussian

export PARDISOLICMESSAGE=1
export OMP_NESTED=TRUE

# LAUNCH 10 MPI processes with x threads each. 8 or 16 threads for larger matrices seems appropriate.
# SEEMS 
# SLURM:
# --ntasks-per-node=
# -N : how nodes
# -n : how many processes per node
# --cpus-per-task=64 : how many threads per task
l1t=1
l2t=1

# machine has 104 cores, so probably 8 x 8 = 64 current best setting. 
# significant increase in performance for pardiso until 16 threads, 32 only slightly faster
export OMP_NUM_THREADS="${l1t},${l2t}"
echo "OMP_NUM_THREADS=${l1t},${l2t}"

#export OMP_NUM_THREADS="${l2t}"
#echo "OMP_NUM_THREADS=${l2t}"

#export MKL_NUM_THREADS=1
#echo "OMP_NUM_THREADS = ${omp_threads}"

#folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_ntFit${ntFit}_ntPred0_noPerTs${noPerTs}_nss${nss}_nb${nb}
folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_nt${nt}_nb${nb}
#folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_nt${nt}
#folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/GaussianData/nb${nb}_no${no}

#source ~/env/cholmod.sh
source ~/.profile

echo "srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type}" 
srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type} >INLA_${solver_type}_output_ns${ns}_ntFit${nt}_ntPred0_nss${nss}_nb${nb}_${num_ranks}_${l1t}_${l2t}_test.txt
#srun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >INLA_RGF_output_ns${ns}_nt${nt}_nb${nb}_${num_ranks}_${l1t}_${l2t}_singleCopyV.txt
#likwid-perfctr -C S0:0-15 -g MEM ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}

