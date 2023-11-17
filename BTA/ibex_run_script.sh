#!/bin/bash

#SBATCH --job-name=RGF		   	#Your Job Name
#SBATCH --nodes=1 			#Number of Nodes desired e.g 1 node
#SBATCH --mem=16G
#SBATCH --gres=gpu:1 			#Run on 1 GPU of any type
#SBATCH --time=00:01:00 		#Walltime: Duration for the Job to run HH:MM:SS
#SBATCH --cpus-per-task=1
#SBATCH --constraint=[v100]
#SBATCH --error=output_RGF.err 		#The .error file name
#SBATCH --output=output_RGF.out 	#The .output file name
### #SBATCH --exclusive

#ns=$1
#nt=$2
#no=$3

ns=492
nt=16
no=7872

#folder_path=/home/x_gaedkelb/RGF/data/ns${ns}
folder_path=/home/x_gaedkelb/b_INLA/data/synthetic/ns${ns}_nt${nt}

nb=2
year=2019

srun nvidia-smi

echo "srun --gres=gpu:1 main ${folder_path} ${ns} ${nt} ${nb} ${no} >RGF_output.txt"
#srun --gres=gpu:v100:1 main ${folder_path} ${ns} ${nt} ${nb} ${folder_path_data} ${no} >${folder_path}/RGF_output.txt
srun ./main ${folder_path} ${ns} ${nt} ${nb} ${folder_path_data} ${no} >RGF_output.txt

srun nvidia-smi
#mv ${folder_path}/L_factor_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}.dat ${folder_path}/L_factor_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_gpu0.dat
#mv ${folder_path_data}/x_sol_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}.dat ${folder_path_data}/x_sol_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}_gpu0.dat
#mv ${folder_path_data}/log_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}.dat ${folder_path_data}/log_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}_gpu0.dat

