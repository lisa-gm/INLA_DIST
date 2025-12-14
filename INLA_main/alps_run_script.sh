#!/bin/bash -l
#SBATCH --job-name=INLAdist
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --account=sm96
#SBATCH --gpus-per-task=1
#SBATCH --partition=debug
####SBATCH --partition=normal
#SBATCH --error=output_INLAdist.err 		#The .error file name
#SBATCH --output=output_INLAdist.out 	#The .output file name

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# threads=$SLURM_CPUS_PER_TASK

BASEPATH=/users/lgaedkem

num_ranks=1

#ns=0
#nss=0
#ntFit=0
#ntPred=0
#nt=$((${ntFit}+${ntPred}))
#nb=6
#no=200

ns=267
nss=0
ntFit=150
ntPred=0
nt=$((${ntFit}+${ntPred}))
nb=3
noPerTs=$((2*${ns}))
#no=$((2*${ns}*${ntFit}))
no=400500

# ns=492
#nss=1442
#ns=20252
#ns=16002
# ntFit=100
# nss=0
#ns=642
#ntFit=250
#ntPred=0
# nt=$((${ntFit}+${ntPred}))
#nss=642
#nt=30
#nb=2
#nb=6
#no=7872
# no=$((2*${ns}*${ntFit}))
# noPerTs=$((2*${ns}))
#no=126

#solver_type=$1
#solver_type=PARDISO
solver_type=BTA
#solver_type=Eigen

data_type=synthetic
#data_type=regression
	
#likelihood=gaussian
likelihood=poisson

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
folder_path=$BASEPATH/data/${data_type}/${likelihood}/ns${ns}_nt${nt}_nb${nb}
#folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_nt${nt}
#folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/GaussianData/nb${nb}_no${no}

#source ~/env/cholmod.sh
source ~/.profile

echo "srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type}" 
srun -n ${num_ranks} ./call_INLA_sliding_windows ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type} >INLA_output_moving_windows_ns${ns}_nt${nt}_nb${nb}_${likelihood}_${solver_type}_${num_ranks}_${l1t}_${l2t}.txt
#srun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >INLA_RGF_output_ns${ns}_nt${nt}_nb${nb}_${num_ranks}_${l1t}_${l2t}_singleCopyV.txt
#likwid-perfctr -C S0:0-15 -g MEM ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}

#srun -n ${num_ranks} nsys profile -o nsys_output_ns${ns}_nt${nt}_${nb}_${num_ranks}_${l1t}_${l2t}_%h%p --stats=true ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type} >INLA_${solver_type}_output_ns${ns}_ntFit${nt}_ntPred0_nss${nss}_nb${nb}_${solver_type}_${num_ranks}_${l1t}_${l2t}_nsys.txt

#./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type} >INLA_${solver_type}_output_ns${ns}_ntFit${nt}_ntPred0_nss${nss}_nb${nb}_${solver_type}_${num_ranks}_${l1t}_${l2t}_test.txt

