#!/bin/bash

#SBATCH --job-name=stdTests           #Your Job Name
#SBATCH --nodes=1                   #Number of Nodes desired e.g 1 nodea
#SBATCH --time=00:59:00                 #Walltime: Duration for the Job to run HH:MM:SS
#SBATCH --gres=gpu:a100:8
#SBATCH --partition=a100
###SBATCH --constraint=a100_80
###SBATCH --qos=a100multi
#SBATCH --exclusive
#SBATCH --error=%x.err          #The .error file name
#SBATCH --output=%x.out         #The .output file name

base_path=/home/hpc/ihpc/ihpc060h/b_INLA/develop
results_folder=${base_path}/standard_tests_new

if [ ! -d ${results_folder} ]; then
  mkdir -p ${results_folder};
fi

num_ranks=9
solver_type=BTA
#solver_type=PARDISO

export PARDISOLICMESSAGE=1
export OMP_NESTED=TRUE

echo "running test script. num ranks = ${num_ranks}. solver type = ${solver_type}."

l1t=2
l2t=1
export OMP_NUM_THREADS="${l1t},${l2t}"
echo "OMP_NUM_THREADS=${l1t},${l2t}"

RESULT_FILE=${results_folder}/results_tests.txt

################################## regression test ##########################################
echo " "

ns=0
nss=0
ntFit=0
ntPred=0
nt=$((${ntFit}+${ntPred}))
nb=6
no=200

data_type=regression
folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_nt${nt}_nb${nb}
test1_output=${results_folder}/INLA_testCase_I_ns${ns}_ntFit${nt}_ntPred0_nss${nss}_nb${nb}_no${no}_${num_ranks}_${l1t}_${l2t}_${solver_type}.txt


echo "TEST CASE I. Spatial-Temporal model. ns = ${ns}, nss = ${nss}, nt = ${nt}, nb = ${nb}, no = ${nb}."

echo "srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${noPerTs} ${folder_path} ${solver_type}" 
srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${folder_path} ${solver_type} >${test1_output}

## theta INLA : 1.33111849377324
## fixed effects INLA : -1.21, 1.23, -0.36, -1.07, -0.156, 1.91

################################ WRITE OUT RESULTS #####################################

echo -e "numRanks numThreadsL1 numThreadsL2 SolverType "  >> ${RESULT_FILE}
echo -e "${num_ranks} ${l1t} ${l2t} ${solver_type} " >> ${RESULT_FILE}
echo -e " " >> ${RESULT_FILE}
echo -e "TEST CASE I. regression model. Gaussian Data. ns = ${ns}, nt = ${nt}, nb = ${nb}, no = ${nb}. " >> ${RESULT_FILE}
echo -n "est.  mean interpret. param. : "  >> ${RESULT_FILE}
cat ${test1_output} | grep "est.  mean interpret. param." | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "mean interpret. param. INLA  : 1.33111849377324 " >> ${RESULT_FILE}
echo -n "est.  mean parameters        : "  >> ${RESULT_FILE}
cat ${test1_output} | grep "est.  mean parameters" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "mean parameters INLA         : 1.33111849377324 " >> ${RESULT_FILE}
echo -n "estimated mean fixed effects : "  >> ${RESULT_FILE}
cat ${test1_output} | grep "estimated mean fixed effects" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "mean fixed effects INLA     : -1.21, 1.23, -0.36, -1.07, -0.156, 1.91" >> ${RESULT_FILE}

################################## TEST III ##########################################
echo " "

ns=492
nss=0
ntFit=50
ntPred=0
nt=$((${ntFit}+${ntPred}))
nb=6
no=$((2*${ns}*${nt}))

data_type=synthetic
folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_nt${nt}_nb${nb}
test3_output=${results_folder}/INLA_testCase_III_ns${ns}_ntFit${nt}_ntPred0_nss${nss}_nb${nb}_${num_ranks}_${l1t}_${l2t}_${solver_type}.txt


echo "TEST CASE III. Spatial-Temporal model. ns = ${ns}, nss = ${nss}, nt = ${nt}, nb = ${nb}, no = ${nb}."

echo "srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${noPerTs} ${folder_path} ${solver_type}" 
srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${folder_path} ${solver_type} >${test1_output}

## orig. mean parameters        :  1.38629400 -5.88254100  1.03972100  3.68887900
## orig. mean interpret. param. : 1.38629400 -0.00000023 2.30258418 1.40625832
## orig. mean fixed effects     :  -1, 3, 0.5, 1, -2



################################ WRITE OUT RESULTS #####################################

echo -e "numRanks numThreadsL1 numThreadsL2 SolverType "  >> ${RESULT_FILE}
echo -e "${num_ranks} ${l1t} ${l2t} ${solver_type} " >> ${RESULT_FILE}
echo -e " " >> ${RESULT_FILE}
echo -e "TEST CASE I. Spatial-Temporal model. ns = ${ns}, nt = ${nt}, nb = ${nb}, no = ${nb}. " >> ${RESULT_FILE}
echo -n "est.  mean interpret. param. : "  >> ${RESULT_FILE}
cat ${test1_output} | grep "est.  mean interpret. param." | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "orig. mean interpret. param. : 1.38629400 -0.00000023 2.30258418 1.40625832" >> ${RESULT_FILE}
echo -n "est.  mean parameters        : "  >> ${RESULT_FILE}
cat ${test1_output} | grep "est.  mean parameters" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "orig. mean parameters        :  1.38629400 -5.88254100  1.03972100  3.68887900" >> ${RESULT_FILE}
echo -n "estimated mean fixed effects : "  >> ${RESULT_FILE}
cat ${test1_output} | grep "estimated mean fixed effects" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "orig. mean fixed effects     :  -1, 3, 0.5, 1, -2" >> ${RESULT_FILE}


################################## TEST IV ##########################################
echo " "

ns=1442
nss=1442
ntFit=60
ntPred=0
nt=$((${ntFit}+${ntPred}))
nb=4
noPerTs=$((2*${ns}))
no=$((2*${ns}*${nt}))

data_type=synthetic
folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_ntFit${ntFit}_ntPred0_noPerTs${noPerTs}_nss${nss}_nb${nb}
test4_output=${results_folder}/INLA_testCase_II_ns${ns}_ntFit${nt}_ntPred0_nss${nss}_nb${nb}_${num_ranks}_${l1t}_${l2t}_${solver_type}.txt

echo "TEST CASE IV. Spatial-Temporal model with add. spatial field. ns = ${ns}, nss = ${nss}, nt = ${nt}, nb = ${nb}, no = ${no}."

echo "srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${noPerTs} ${folder_path} ${solver_type}" 
srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${folder_path} ${solver_type} >${test2_output}

## interpret.param 1.386 0.405 1.099 1.386 -1.204 1.099
## theta.original  1.386 -4.42 0.634 1.674 -4.608 2.244
## fixed effects   -1 3 0.5 2


################ WRITE OUT RESULTS #################

echo -e "numRanks numThreadsL1 numThreadsL2 SolverType "  >> ${RESULT_FILE}
echo -e "${num_ranks} ${l1t} ${l2t} ${solver_type} " >> ${RESULT_FILE}
echo -e " " >> ${RESULT_FILE}
echo -e "TEST CASE IV. Spatial-Temporal model with add. spatial-field. ns = ${ns}, nt = ${nt}, nb = ${nb}, no = ${nb}. " >> ${RESULT_FILE}
echo -n "est.  mean interpret. param. : "  >> ${RESULT_FILE}
cat ${test2_output} | grep "est.  mean interpret. param." | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "orig. mean interpret. param. : 1.386 0.405 1.099 1.386 -1.204 1.099" >> ${RESULT_FILE}
echo -n "est.  mean parameters        : "  >> ${RESULT_FILE}
cat ${test2_output} | grep "est.  mean parameters" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "orig. mean parameters        :  1.386 -4.42 0.634 1.674 -4.608 2.244" >> ${RESULT_FILE}
echo -n "estimated mean fixed effects : "  >> ${RESULT_FILE}
cat ${test2_output} | grep "estimated mean fixed effects" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "orig. mean fixed effects     :  -1 3 0.5 2" >> ${RESULT_FILE}
