# hard code example to see if code runs

BASE_PATH=/home/x_gaedkelb
#BASE_PATH=/home/hpc/ihpc/ihpc060h

#ns=$1
#nt=$2
#nb=$3
#no=$4

ns=641
nt=15
nb=3
no=31196

#data_type=synthetic
data_type=temperature

# number of aggregated days
# k=7

#solver_type=$1
solver_type=PARDISO
#solver_type=RGF
	
export PARDISOLICMESSAGE=1
export OMP_NESTED=TRUE

num_ranks=$1

# TO KEEP IN MIND:
# l2t also manually set in pardiso. l2t needs to be set a smaller or equal value to the min value in pardiso.
# And it seems like l1t needs to be smaller or equal to l2t. I don't know why.

# nested parallelism : 
l1t=2
# 2nd number : Pardiso will be run with this many threads for each linear system
l2t=4
# significant increase in performance for pardiso until 16 threads, 32 only slightly faster
export OMP_NUM_THREADS="${l1t},${l2t}"
echo "OMP_NUM_THREADS=${l1t},${l2t}"

#export MKL_NUM_THREADS=1
#echo "OMP_NUM_THREADS = ${omp_threads}"

#folder_path=${BASE_PATH}/b_INLA/data/${data_type}/ns${ns}_nt${nt}
folder_path=${BASE_PATH}/b_INLA/data/${data_type}/ns${ns}_nt${nt}_nb${nb}
#folder_path=/home/x_gaedkelb/b_INLA/data/spatial_toy_ex
#folder_path=/home/x_gaedkelb/b_INLA/data



echo "mpirun -np ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}" 
#srun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} #>${folder_path}/output_${solver_type}_${ns}_${nt}.txt
mpirun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} #>${folder_path}/output_${solver_type}_${ns}_${nt}.txt
#likwid-mpirun -np ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} #>${folder_path}/output_${solver_type}_${ns}_${nt}.txt

##likwid-perfctr -C S0:0-15 -g MEM ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}
