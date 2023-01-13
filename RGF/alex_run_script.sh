#!/bin/bash

#SBATCH --job-name=RGF		   	#Your Job Name
#SBATCH --nodes=1 			#Number of Nodes desired e.g 1 node
#SBATCH --gres=gpu:a40:1 			#Run on 1 GPU of any type
#SBATCH --time=00:01:00 		#Walltime: Duration for the Job to run HH:MM:SS
#####SBATCH --cpus-per-task=1
#SBATCH --error=output_RGF.err 		#The .error file name
#SBATCH --output=output_RGF.out 	#The .output file name
#####SBATCH --exclusive

#ns=$1
#nt=$2
#no=$3

ns=492
nt=16
nb=6
no=15744

folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/synthetic/ns${ns}_nt${nt}_nb${nb}

solver_type=RGF
##year=2019


echo "srun --gres=gpu:1 main ${folder_path} ${ns} ${nt} ${nb} ${no} >RGF_output.txt"
# main.C
#srun ./main ${folder_path} ${ns} ${nt} ${nb} ${folder_path_data} ${no} >RGF_output.txt

# mainEigen.C -> input: ns nt nb no path/to/files solver_type
srun ./mainEigen ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >RGF_output.txt

#mv ${folder_path_data}/log_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}.dat ${folder_path_data}/log_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}_gpu0.dat

## ./mainEigen 492 16 6 15744 /home/hpc/ihpc/ihpc060h/b_INLA/data/synthetic/ns492_nt16_nb6 RGF