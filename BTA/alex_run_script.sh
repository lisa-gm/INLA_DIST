#!/bin/bash

#SBATCH --job-name=BTA		   	#Your Job Name
#SBATCH --nodes=1 			#Number of Nodes desired e.g 1 node
#SBATCH --gres=gpu:a40:1 			#Run on 1 GPU of any type
###SBATCH --gres=gpu:a100:1                        #Run on 1 GPU of any type
#SBATCH --time=01:01:00 		#Walltime: Duration for the Job to run HH:MM:SS
#####SBATCH --cpus-per-task=1
####SBATCH --constraint=a100_80
#SBATCH --error=output_BTA.err 		#The .error file name
#SBATCH --output=output_BTA.out 	#The .output file name
###SBATCH --exclusive

#ns=2307
#ns=2865
ns=492
nss=0
nt=100
#nt=365
#nb=4
nb=6
#no=200
#nss=642

#ns=1442
#nss=1442
#nt=60
#nb=4
#no=15744
#no=3885561
#no=2472561
no=$((2*${ns}*${nt}))
#noPerTs=$((2*${ns}))

#data_type=regression
data_type=synthetic
#data_type=temperature

#folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/GaussianData/nb${nb}_no${no}
folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_nt${nt}_nb${nb}
#folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/synthetic/fixed_ns${ns}_nt${nt}_nb${nb}
#folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_ntFit${nt}_ntPred0_noPerTs${noPerTs}_nss${nss}_nb${nb}


solver_type=BTA
#year=2019

#gpuSolver=MAGMA
gpuSolver=MAGMA_EXPERT
#gpuSolver=CUDA


#precision=double
#precision=single

threads=1
export OMP_NUM_THREADS=${threads}
echo "OMP_NUM_THREADS=${threads}"

export CUDA_LAUNCH_BLOCKING=1
echo "CUDA_LAUNCH_BLOCKING=1"


echo "./mainEigen ${ns} ${nt} ${nss} ${nb} ${no} ${folder_path} ${solver_type} >BTA_output.txt"
# main.C
#srun ./main ${folder_path} ${ns} ${nt} ${nb} ${folder_path_data} ${no} >BTA_output.txt

# mainEigen.C -> input: ns nt nb no path/to/files solver_type
#srun nsys profile -o nsys_output_magma_gpuOnly_%h_%p.txt ./mainEigen ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >BTA_output_magma_gpuOnly_ns${ns}_nt${nt}_nb${nb}_${threads}.txt
#srun nsys profile -o nsys_output_cuda_seq_ns20252_nt30_nb6_%h_%p.txt ./mainEigen ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >BTA_output_cuda_seq_ns${ns}_nt${nt}_nb${nb}_${threads}.txt
#srun nsys profile -o nsys_output_cuda_seq_ns1442_nt60_nss1442_nb4_%h_%p.txt ./mainEigen ${ns} ${nt} ${nss} ${nb} ${no} ${folder_path} ${solver_type} >BTA_output_cuda_seq_getFlOPS_test_selInv_cudaTRTRI_ns${ns}_nt${nt}_nss${nss}_nb${nb}_${threads}.txt
#srun nsys profile -o nsys_output_cuda_ns1442_nt60_nss1442_nb4_%h_%p ./mainEigen ${ns} ${nt} ${nss} ${nb} ${no} ${folder_path} ${solver_type} >BTA_output_cudaTRTRI_ns${ns}_nt${nt}_nss${nss}_nb${nb}_${threads}.txt

#srun nsys profile -o nsys_output_%h_%p.txt ./mainEigen ${ns} ${nt} ${nss} ${nb} ${no} ${folder_path} ${solver_type} >BTA_output_ns${ns}_nt${nt}_nb${nb}_${threads}.txt
srun ./mainEigen ${ns} ${nt} ${nss} ${nb} ${no} ${folder_path} ${solver_type} >BTA_cudaGEMM_${gpuSolver}_ns${ns}_nt${nt}_nss${nss}_nb${nb}_${threads}_1_i2.txt

#mv ${folder_path_data}/log_BTA_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}.dat ${folder_path_data}/log_BTA_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}_gpu0.dat

## ./mainEigen 492 16 6 15744 /home/hpc/ihpc/ihpc060h/b_INLA/data/synthetic/ns492_nt16_nb6 BTA
## ./mainEigen 492 50 6 49200  /home/hpc/ihpc/ihpc060h/b_INLA/data/synthetic/ns492_nt50_nb6 BTA
## ./mainEigen 492 500 6 492000  /home/hpc/ihpc/ihpc060h/b_INLA/data/synthetic/ns492_nt500_nb6 BTA
