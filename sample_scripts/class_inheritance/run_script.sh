# run_script


ns=436
nt=16
nb=2
no=4057
solver_type=$1

export PARDISOLICMESSAGE=1

export OMP_NESTED=TRUE

# TO KEEP IN MIND:
# l2t also manually set in pardiso. l2t needs to be set a smaller or equal value to the min value in pardiso.
# And it seems like l1t needs to be smaller or equal to l2t. I don't know why.

# nested parallelism : 
# 1st number : 8 because dim(theta)=4, forward & backward difference in gradient
l1t=8
# 2nd number : Pardiso will be run with this many threads for each linear system
l2t=8
# machine has 104 cores, so probably 8 x 8 = 64 current best setting. 
# significant increase in performance for pardiso until 16 threads, 32 only slightly faster
export OMP_NUM_THREADS="${l1t},${l2t}"
echo "OMP_NUM_THREADS=${l1t},${l2t}"

#export MKL_NUM_THREADS=1
#echo "OMP_NUM_THREADS = ${omp_threads}"

folder_path=/home/x_gaedkelb/b_INLA/data/temperature/ns${ns}_nt${nt}
#folder_path=/home/x_gaedkelb/b_INLA/data/synthetic/ns${ns} #_nt${nt}


echo "./main ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}"
./main ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}