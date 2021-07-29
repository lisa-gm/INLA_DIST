# hard code example to see if code runs

#ns=2252
#nb=2
#no=11646

#ns=$1
#nt=$2
#nb=$3
#no=$4

#ns=1002
#nt=16
#nb=2
#no=188242

ns=492
nt=16
nb=2
no=7872
	
export PARDISOLICMESSAGE=1

export OMP_NESTED=TRUE
# nested parallelism : 
# 1st number : 8 because dim(theta)=4, forward & backward difference in gradient
l1t=1
# 2nd number : Pardiso will be run with this many threads for each linear system
l2t=16
# machine has 104 cores, so probably 8 x 8 = 64 current best setting. 
# significant increase in performance for pardiso until 16 threads, 32 only slightly faster
export OMP_NUM_THREADS="${l1t},${l2t}"
echo "OMP_NUM_THREADS=${l1t},${l2t}"

#export OMP_NUM_THREADS=${omp_threads}
#echo "OMP_NUM_THREADS = ${omp_threads}"

#folder_path=/home/x_gaedkelb/b_INLA/data/temperature/ns${ns}_nt${nt}
folder_path=/home/x_gaedkelb/b_INLA/data/synthetic/ns${ns}_nt${nt}

echo "./example ${ns} ${nt} ${nb} ${no} ${folder_path}" 
./example ${ns} ${nt} ${nb} ${no} ${folder_path}
