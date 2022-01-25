#!/bin/bash

#SBATCH --job-name=call_INLA   	#Your Job Name
#SBATCH --nodes=1 			#Number of Nodes desired e.g 1 node
#SBATCH --time=00:01:00 		#Walltime: Duration for the Job to run HH:MM:SS
#SBATCH --mail-user=useremail@kaust.edu.sa #Your Email address assigned for your job
#SBATCH --mail-type=ALL 		#Receive an email for ALL Job Statuses
#SBATCH --error=output_call_INLA.err 		#The .error file name
#SBATCH --output=output_call_INLA.out 	#The .output file name


# hard code example to see if code runs

# Spatial model toy dataset
#ns=490
#nt=0
#nb=1
#no=200

# synthetic dataset
ns=492
nt=16
nb=2
no=7872

#ns=$1
#nt=$2
#nb=$3
#no=$4

#ns=1002
#nt=16
#nb=2
#no=188242

# temperature datasets
#ns=42
#ns=436
#nt=0
#nt=16
#nt=52
#nt=20
#nb=2
#no=11646
#no=4057
#no=12873
#no=5148

data_type=synthetic
#data_type=temperature

# number of aggregated days
#k=4

#solver_type=$1
solver_type=PARDISO


	
export PARDISOLICMESSAGE=1

export OMP_NESTED=TRUE

# TO KEEP IN MIND:
# l2t also manually set in pardiso. l2t needs to be set a smaller or equal value to the min value in pardiso.
# And it seems like l1t needs to be smaller or equal to l2t. I don't know why.

# nested parallelism : 
# 1st number : 8 because dim(theta)=4, forward & backward difference in gradient
l1t=9
# 2nd number : Pardiso will be run with this many threads for each linear system
l2t=16
# machine has 104 cores, so probably 8 x 8 = 64 current best setting. 
# significant increase in performance for pardiso until 16 threads, 32 only slightly faster
export OMP_NUM_THREADS="${l1t},${l2t}"
echo "OMP_NUM_THREADS=${l1t},${l2t}"

#export MKL_NUM_THREADS=1
#echo "OMP_NUM_THREADS = ${omp_threads}"

#folder_path=/home/x_gaedkelb/b_INLA/data/${data_type}/ns${ns}_nt${nt}_k${k}
folder_path=/home/x_gaedkelb/b_INLA/data/${data_type}/ns${ns}_nt${nt}
#folder_path=/home/x_gaedkelb/b_INLA/data/spatial_toy_ex
#folder_path=/home/x_gaedkelb/b_INLA/data



echo "srun call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >RGF_output.txt" 
srun ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >RGF_output.txt
#likwid-perfctr -C S0:0-15 -g MEM ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}

