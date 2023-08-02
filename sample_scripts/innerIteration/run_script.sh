### run script
num_ranks=1

BASE_PATH=/home/x_gaedkelb
data_type=synthetic

n=3
m=50

#ns=92
#nt=40
#nss=92
#nb=2
#noPerTs=$((2*${ns}))
#no=$((2*${ns}*${nt}))

#folder_path=${BASE_PATH}/b_INLA/data/${data_type}/ns${ns}_nt${nt}_nb${nb}
#folder_path=${BASE_PATH}/b_INLA/data/${data_type}/ns${ns}_ntFit${nt}_ntPred0_noPerTs${noPerTs}_nss${nss}_nb${nb}
folder_path=${BASE_PATH}/b_INLA/sample_scripts/innerIteration/PoissonData

solver_type=BTA

echo "./main ${ns} ${nt} ${nss} ${nb} ${no} ${folder_path} ${solver_type}"
#mpirun -n ${num_ranks} ./main ${ns} ${nt} ${nss} ${nb} ${no} ${folder_path} ${solver_type} 

mpirun -n ${num_ranks} ./main ${n} ${m} ${folder_path}
