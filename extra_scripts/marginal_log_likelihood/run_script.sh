### run script
num_ranks=1

#BASE_PATH=/home/x_gaedkelb
BASE_PATH=/home/hpc/ihpc/ihpc060h

data_type=synthetic

ns=812
nt=50
nss=0
nb=6
noPerTs=$((2*${ns}))
no=$((2*${ns}*${nt}))

folder_path=${BASE_PATH}/b_INLA/data/${data_type}/ns${ns}_nt${nt}_nb${nb}
#folder_path=${BASE_PATH}/b_INLA/data/${data_type}/ns${ns}_ntFit${nt}_ntPred0_noPerTs${noPerTs}_nss${nss}_nb${nb}

solver_type=PARDISO

num_threads=32
echo "export OMP_NUM_THREADS=${num_threads}"
export OMP_NUM_THREADS=${num_threads}

export PARDISOLICMESSAGE=1


echo "./main ${ns} ${nt} ${nss} ${nb} ${no} ${folder_path} ${solver_type}"
mpirun -n ${num_ranks} ./main ${ns} ${nt} ${nss} ${nb} ${no} ${folder_path} ${solver_type} 
