### run script
num_ranks=1

BASE_PATH=/home/x_gaedkelb
data_type=synthetic
#data_type=regression


# ns=436
# nt=0
# nss=0
# nb=8
# no=1308

ns=335
nt=15
nss=0
nb=10
no=15075

# ns=347
# nt=0
# nss=0
# nb=5
#noPerTs=$((2*${ns}))
#no=$((2*${ns}*${nt}))
# no=734
#no=14680

likelihood=Poisson
#likelihood=Binomial

#folder_path=${BASE_PATH}/b_INLA/data/${data_type}/${likelihood}Data/nb${nb}_no${no}
folder_path=${BASE_PATH}/b_INLA/data/${data_type}/${likelihood}Data/ns${ns}_nt${nt}_nss${nss}_nb${nb}_no${no}
#folder_path=${BASE_PATH}/b_INLA/data/${data_type}/${likelihood}Data/ns${ns}_nt${nt}_nb${nb}_no${no}
#folder_path=${BASE_PATH}/b_INLA/data/${data_type}/ns${ns}_ntFit${nt}_ntPred0_noPerTs${noPerTs}_nss${nss}_nb${nb}
#folder_path=${BASE_PATH}/b_INLA/sample_scripts/innerIteration/${likelihood}

solver_type=BTA

export OMP_NUM_THREADS=16

#echo "mpirun -n ${num_ranks} ./main ${n} ${m} ${likelihood} ${folder_path}"
#mpirun -n ${num_ranks} ./main ${n} ${m} ${likelihood} ${folder_path}

echo "./main ${ns} ${nt} ${nss} ${nb} ${no} ${folder_path} ${solver_type}"
mpirun -n ${num_ranks} ./main ${ns} ${nt} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type} 



