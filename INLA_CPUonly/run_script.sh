# hard code example to see if code runs

timestamp=$(date +%s)

#BASE_PATH=/home/x_gaedkelb
BASE_PATH=/home/hpc/ihpc/ihpc060h

# synthetic dataset
ns=492
nt=100
nb=6


# ns=92
# nt=5
# nb=6
no=$((2*${ns}*${nt}))

data_type=synthetic
#data_type=temperature


#solver_type=$1
solver_type=PARDISO
#solver_type=BTA
	
export PARDISOLICMESSAGE=1
#export PARDISO_WRITE_MAT=1

num_ranks=1

#################### NUM THREADS
# l1t=1
# l2t=16
# significant increase in performance for pardiso until 16 threads, 32 only slightly faster

# Configure nested OpenMP parallelism using modern approach
export OMP_NUM_THREADS=16
echo "OMP_NUM_THREADS = ${OMP_NUM_THREADS}"
# export OMP_MAX_ACTIVE_LEVELS=2
# export OMP_NUM_THREADS="${l1t},${l2t}"
# export OMP_PLACES=cores
# export OMP_PROC_BIND=spread,close
# echo "OpenMP Configuration:"
# echo "  OMP_MAX_ACTIVE_LEVELS=${OMP_MAX_ACTIVE_LEVELS}"
# echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
# echo "  OMP_PLACES=${OMP_PLACES}"
# echo "  OMP_PROC_BIND=${OMP_PROC_BIND}"


############### FOLDER PATH 
#folder_path=${BASE_PATH}/b_INLA/data/${data_type}/ns${ns}_nt${nt}
folder_path=${BASE_PATH}/data/${data_type}/ns${ns}_nt${nt}_nb${nb}
#folder_path=/home/x_gaedkelb/b_INLA/data/spatial_toy_ex
#folder_path=/home/x_gaedkelb/b_INLA/data

#export LD_LIBRARY_PATH=${HOME}/applications/pardiso_iterative:$LD_LIBRARY_PATH

folder_path=${HOME}/data/${data_type}/ns${ns}_nt${nt}_nb${nb}
#folder_path=/users/lgaedkem/b_INLA/data/${data_type}/ns${ns}_nt${nt}
# taskset -c 0-36

# CAREFUL : needs to be AT LEAST 11 (main + 10 workers, 10 because of hessian, for BFGS only 9 are required)
echo "srun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}" 
srun -n ${num_ranks}  ./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >INLA_PARDISO_ns${ns}_nt${nt}_nb${nb}_${num_ranks}_${l1t}_${l2t}_Pardiso_3IterV.txt
#./call_INLA ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >INLA_PARDISO_ns${ns}_nt${nt}_nb${nb}_${num_ranks}_${l1t}_${l2t}_Pardiso_3IterV.txt
