# hard code example to see if code runs


BASE_PATH=/home/x_gaedkelb
#BASE_PATH=/home/hpc/ihpc/ihpc060h

ns=492
nt=500
nb=6
no=$((2*${ns}*${nt}))

data_type=synthetic
#data_type=temperature

# number of aggregated days
# k=7

#solver_type=$1
solver_type=PARDISO
#solver_type=RGF
	
export PARDISOLICMESSAGE=1
export OMP_NESTED=TRUE

# TO KEEP IN MIND:
# l2t also manually set in pardiso. l2t needs to be set a smaller or equal value to the min value in pardiso.
# And it seems like l1t needs to be smaller or equal to l2t. I don't know why.

# nested parallelism : 
l1t=2
# 2nd number : Pardiso will be run with this many threads for each linear system
l2t=8
# significant increase in performance for pardiso until 16 threads, 32 only slightly faster
export OMP_NUM_THREADS="${l1t},${l2t}"
echo "OMP_NUM_THREADS=${l1t},${l2t}"

#export MKL_NUM_THREADS=8
#echo "OMP_NUM_THREADS = ${omp_threads}"

folder_path=${BASE_PATH}/b_INLA/data/${data_type}/ns${ns}_nt${nt}_nb${nb}



for i in 1
do       
	no=$((2*${ns}*${nt}))
	folder_path=${BASE_PATH}/b_INLA/data/${data_type}/ns${ns}_nt${nt}_nb${nb}
	echo "./main ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}" 
	./main ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >output_old${solver_type}_${ns}_${nt}_${nb}_${l1t}_${l2t}.txt

	#echo "./main_standalone ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}" 
	#./main_standalone ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} #>output_measure_variance_${solver_type}_${ns}_${nt}_${nb}_${num_ranks}_${l1t}_${l2t}_l2test_i${i}.txt


done
