#!/bin/bash

#SBATCH --job-name=batch_call_INLA_RGF_2           #Your Job Name
#SBATCH --nodes=1                       #Number of Nodes desired e.g 1 node
#SBATCH --time=05:59:00                 #Walltime: Duration for the Job to run HH:MM:SS
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=a100
######SBATCH --exclusive
#SBATCH --error=%x.err          #The .error file name
#SBATCH --output=%x.out         #The .output file name

num_ranks=2

#ns=1002
ns=10242
nb=6

solver_type=RGF

data_type=synthetic

export PARDISOLICMESSAGE=1
export OMP_NESTED=TRUE

l1t=2
l2t=1

export OMP_NUM_THREADS="${l1t},${l2t}"
echo "OMP_NUM_THREADS=${l1t},${l2t}"

for nt in 200 #100 200 500 1000 2000 
do
	no=$((2*${ns}*${nt}))
	echo ${no}
	folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_nt${nt}_nb${nb}
	echo "srun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}"
	srun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >${folder_path}/INLA_RGF_2_output_ns${ns}_nt${nt}_nb${nb}.txt
done

