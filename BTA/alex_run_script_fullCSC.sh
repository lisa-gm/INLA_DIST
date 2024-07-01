#!/bin/bash

#SBATCH --job-name=RGF		   	#Your Job Name
#SBATCH --nodes=1 			#Number of Nodes desired e.g 1 node
#SBATCH --gres=gpu:a100:1 			#Run on 1 GPU of any type
#SBATCH --time=01:01:00 		#Walltime: Duration for the Job to run HH:MM:SS
#####SBATCH --cpus-per-task=1
####SBATCH --constraint=a100_80
#SBATCH --error=output_RGF.err 		#The .error file name
#SBATCH --output=output_RGF.out 	#The .output file name
#####SBATCH --exclusive

ns=642
nt=60
#no=$3
nss=642

#ns=4002
#nt=250
nb=4
#no=15744
no=$((2*${ns}*${nt}))
noPerTs=$((2*${ns}))

data_type=synthetic

#folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/synthetic/ns${ns}_nt${nt}_nb${nb}
#folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/synthetic/fixed_ns${ns}_nt${nt}_nb${nb}
folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_ntFit${nt}_ntPred0_noPerTs${noPerTs}_nss${nss}_nb${nb}


solver_type=BTA
##year=2019

threads=1
export OMP_NUM_THREADS=${threads}
echo "OMP_NUM_THREADS=${threads}"

#export CUDA_LAUNCH_BLOCKING=1
#echo "CUDA_LAUNCH_BLOCKING=1"


echo "srun --gres=gpu:1 main ${folder_path} ${ns} ${nt} ${nss} ${nb} ${no} >RGF_output.txt"
srun ./main_fullCSC ${ns} ${nt} ${nss} ${nb} ${no} ${folder_path} ${solver_type} >BTA_output_fullCSC_ns${ns}_nt${nt}_nss${nss}_nb${nb}_${threads}.txt
