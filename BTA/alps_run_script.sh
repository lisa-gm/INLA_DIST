#!/bin/bash -l
#SBATCH --job-name=BTA_solver
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
####SBATCH --mem=32G              # Request 32 GB total memory
#SBATCH --cpus-per-task=1
#SBATCH --account=sm96
#SBATCH --partition=debug
####SBATCH --partition=normal
#SBATCH --error=output_BTA.err 		#The .error file name
#SBATCH --output=output_BTA.out 	#The .output file name

OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS

ulimit -s unlimited

BASEPATH=/users/lgaedkem

ns=4002
nt=250
#no=$3
nss=0

#ns=4002
#nt=250
nb=6
#no=15744
no=$((2*${ns}*${nt}))
noPerTs=$((2*${ns}))

data_type=synthetic

folder_path=$BASEPATH/data/synthetic/gaussian/ns${ns}_nt${nt}_nb${nb}
#folder_path=$BASEPATH/data/synthetic/fixed_ns${ns}_nt${nt}_nb${nb}
#folder_path=$BASEPATH/data/${data_type}/ns${ns}_ntFit${nt}_ntPred0_noPerTs${noPerTs}_nss${nss}_nb${nb}

solver_type=BTA

# threads=1
# export OMP_NUM_THREADS=${threads}
# echo "OMP_NUM_THREADS=${threads}"

# export CUDA_LAUNCH_BLOCKING=1
# echo "CUDA_LAUNCH_BLOCKING=1"


echo "srun --gres=gpu:1 main ${folder_path} ${ns} ${nt} ${nss} ${nb} ${no} >RGF_output.txt"

# mainEigen.C -> input: ns nt nb no path/to/files solver_type
#srun nsys profile -o nsys_output_magma_gpuOnly_%h_%p.txt ./mainEigen ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >BTA_output_magma_gpuOnly_ns${ns}_nt${nt}_nb${nb}_${threads}.txt
#srun nsys profile -o nsys_output_cuda_seq_ns20252_nt30_nb6_%h_%p.txt ./mainEigen ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >BTA_output_cuda_seq_ns${ns}_nt${nt}_nb${nb}_${threads}.txt
#srun nsys profile -o nsys_output_cuda_seq_ns1442_nt60_nss1442_nb4_%h_%p.txt ./mainEigen ${ns} ${nt} ${nss} ${nb} ${no} ${folder_path} ${solver_type} >BTA_output_cuda_seq_getFlOPS_test_selInv_cudaTRTRI_ns${ns}_nt${nt}_nss${nss}_nb${nb}_${threads}.txt
#srun nsys profile -o nsys_output_cuda_ns1442_nt60_nss1442_nb4_%h_%p.txt ./mainEigen ${ns} ${nt} ${nss} ${nb} ${no} ${folder_path} ${solver_type} >BTA_output_cudaTRTRI_ns${ns}_nt${nt}_nss${nss}_nb${nb}_${threads}.txt

#srun nsys profile -o nsys_output_magma_gpuOnly_seq_%h_%p.txt ./mainEigen ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >BTA_output_magma_gpuOnly_seq_ns${ns}_nt${nt}_nb${nb}_${threads}.txt

srun ./mainEigen ${ns} ${nt} ${nss} ${nb} ${no} ${folder_path} ${solver_type} >BTA_output_ns${ns}_nt${nt}_nss${nss}_nb${nb}_${OMP_NUM_THREADS}.txt

#srun ./eigen_demo 