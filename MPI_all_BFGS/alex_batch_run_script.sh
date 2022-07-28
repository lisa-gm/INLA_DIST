#!/bin/bash

#SBATCH --job-name=batch_call_INLA_RGF           #Your Job Name
#SBATCH --nodes=3                       #Number of Nodes desired e.g 1 node
#SBATCH --time=11:59:00                 #Walltime: Duration for the Job to run HH:MM:SS
#SBATCH --gres=gpu:a100:8
#SBATCH --partition=a100
#SBATCH --qos=a100multi
#SBATCH --exclusive
#SBATCH --error=%x.err          #The .error file name
#SBATCH --output=%x.out         #The .output file name

num_ranks=9

ns=19362
nt=30
#ns=10242
nb=6

solver_type=RGF

data_type=synthetic

export PARDISOLICMESSAGE=1
export OMP_NESTED=TRUE

l1t=2
l2t=1

export OMP_NUM_THREADS="${l1t},${l2t}"
echo "OMP_NUM_THREADS=${l1t},${l2t}"

for i in 1
do

	#for num_ranks in 3 2 1
	#do
		no=$((2*${ns}*${nt}))
		echo "ns = ${ns}, nt = ${nt}, i = ${i}, no = ${no}"
		folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_nt${nt}_nb${nb}
		echo "srun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}"
		srun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >INLA_RGF_output_ns${ns}_nt${nt}_nb${nb}_mpi${num_ranks}_l1t${l1t}_l2t${l2t}_i${i}_singleCopyV.txt
	#done
done

