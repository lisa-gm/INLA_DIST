#!/bin/bash

#SBATCH --job-name=small_test           #Your Job Name
#SBATCH --nodes=1                      #Number of Nodes desired e.g 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00                 #Walltime: Duration for the Job to run HH:MM:SS
#SBATCH --error=small_test.err          #The .error file name
#SBATCH --output=small_test.out         #The .output file name

#num_ranks=1
## mpi run script

#export OMP_NUM_THREADS=4


likwid-perfctr -C S0:0 -g L3 -m ./perfctr_example
#srun ./perfctr_example
