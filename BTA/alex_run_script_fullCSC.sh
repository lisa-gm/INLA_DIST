#!/bin/bash

#SBATCH --job-name=BTA		   	#Your Job Name
#SBATCH --nodes=1 			#Number of Nodes desired e.g 1 node
#SBATCH --gres=gpu:a100:1 			#Run on 1 GPU of any type
#SBATCH --time=01:01:00 		#Walltime: Duration for the Job to run HH:MM:SS
#####SBATCH --cpus-per-task=1
#####SBATCH --constraint=a100_80
#SBATCH --error=output_BTA.err 		#The .error file name
#SBATCH --output=output_BTA.out 	#The .output file name
####SBATCH --exclusive

# ns=2865
# nt=365
# nss=0
# nb=4

# ns=4002
# nt=250
# nss=0
# nb=6

ns=42
nt=3
nss=0
nb=2

n=$((${ns}*${nt}+${nss}+${nb}))

#folder_path=/home/vault/j101df/j101df10/inla_matrices/toy_examples
folder_path=/home/vault/j101df/j101df10/inla_matrices/INLA_paper_examples

Q_file=${folder_path}/Qxy_ns${ns}_nt${nt}_nss${nss}_nb${nb}_n${n}.dat
#Q_file=Qxy_ns${ns}_nt${nt}_nss${nss}_nb${nb}_n${n}.dat

threads=1
export OMP_NUM_THREADS=${threads}
echo "OMP_NUM_THREADS=${threads}"

export CUDA_LAUNCH_BLOCKING=1
echo "CUDA_LAUNCH_BLOCKING=1"

echo "srun --gres=gpu:1 main ${ns} ${nt} ${nss} ${nb} ${Q_file} >BTA_output.txt"
srun ./main_fullCSC ${ns} ${nt} ${nss} ${nb} ${Q_file} #>BTA_output_fullCSC_ns${ns}_nt${nt}_nss${nss}_nb${nb}_selINV_oldV.txt

#srun nsys profile -o nsys_output_fullCSC_ns42_nt3_nb2_%h_%p.txt ./main_fullCSC ${ns} ${nt} ${nss} ${nb} ${Q_file} >BTA_output_nsys_fullCSC_ns${ns}_nt${nt}_nss${nss}_nb${nb}.txt
#srun nsys profile -o nsys_output_fullCSC_ns2865_nt250_nb6_%h_%p.txt ./main_fullCSC ${ns} ${nt} ${nss} ${nb} ${Q_file} >BTA_output_nsys_fullCSC_ns${ns}_nt${nt}_nss${nss}_nb${nb}.txt
#srun nsys profile -o nsys_output_fullCSC_ns2865_nt365_nb4_%h_%p.txt ./main_fullCSC ${ns} ${nt} ${nss} ${nb} ${Q_file} >BTA_output_nsys_fullCSC_ns${ns}_nt${nt}_nss${nss}_nb${nb}.t