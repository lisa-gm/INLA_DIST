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
results_folder=${base_path}/standard_tests_updatedMuStorage

if [ ! -d ${results_folder} ]; then
  mkdir -p ${results_folder};
fi

num_ranks=5

solver_type=BTA
#solver_type=PARDISO

export PARDISOLICMESSAGE=1
export OMP_NESTED=TRUE

echo "running test script. num ranks = ${num_ranks}. solver type = ${solver_type}."

l1t=2
l2t=1
export OMP_NUM_THREADS="${l1t},${l2t}"
echo "OMP_NUM_THREADS=${l1t},${l2t}"

RESULT_FILE=${results_folder}/results_tests_${solver_type}_${num_ranks}_${l1t}_${l2t}.txt

source ~/env/cholmod.sh

##################################
###### GAUSSIAN LIKELIHOOD #######
##################################

################################## TEST I ##########################################
echo " "

ns=0
nss=0
ntFit=0
ntPred=0
nt=$((${ntFit}+${ntPred}))
nb=6
no=200

data_type=regression
likelihood=gaussian
folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/GaussianData/nb${nb}_no${no}
test1_output=${results_folder}/INLA_testCase_I_ns${ns}_ntFit${nt}_ntPred0_nss${nss}_nb${nb}_no${no}_${num_ranks}_${l1t}_${l2t}_${solver_type}.txt

echo "TEST CASE I. regression model. Gaussian Data. ns = ${ns}, nss = ${nss}, nt = ${nt}, nb = ${nb}, no = ${no}."

echo "srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type}" 
srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type} >${test1_output}

## theta INLA : 1.33111849377324
## fixed effects INLA : -1.21, 1.23, -0.36, -1.07, -0.156, 1.91

################################ WRITE OUT RESULTS #####################################

#echo -e " " >> ${RESULT_FILE}
echo -e "TEST CASE I. regression model. Gaussian Data. ns = ${ns}, nt = ${nt}, nb = ${nb}, no = ${nb}. " >> ${RESULT_FILE}
echo -e "numRanks numThreadsL1 numThreadsL2 SolverType "  >> ${RESULT_FILE}
echo -e "${num_ranks} ${l1t} ${l2t} ${solver_type} " >> ${RESULT_FILE}
echo -n "est.  mean interpret. param. : "  >> ${RESULT_FILE}
cat ${test1_output} | grep "est.  mean interpret. param." | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "mean interpret. param. INLA  : 1.33111849377324 " >> ${RESULT_FILE}
echo -n "est.  mean parameters        : "  >> ${RESULT_FILE}
cat ${test1_output} | grep "est.  mean parameters" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "mean parameters INLA         : 1.33111849377324 " >> ${RESULT_FILE}
echo -n "estimated mean fixed effects : "  >> ${RESULT_FILE}
cat ${test1_output} | grep "estimated mean fixed effects" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "mean fixed effects INLA     : -1.21, 1.23, -0.36, -1.07, -0.156, 1.91" >> ${RESULT_FILE}

################################## TEST II ##########################################
echo " "

ns=425
nss=0
ntFit=0
ntPred=0
nt=$((${ntFit}+${ntPred}))
nb=3
no=595

data_type=synthetic
likelihood=gaussian
folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_nt${nt}_nb${nb}
test2_output=${results_folder}/INLA_testCase_II_ns${ns}_ntFit${nt}_ntPred0_nss${nss}_nb${nb}_no${no}_${num_ranks}_${l1t}_${l2t}_${solver_type}.txt

echo "TEST CASE II. spatial model. Gaussian Data. ns = ${ns}, nss = ${nss}, nt = ${nt}, nb = ${nb}, no = ${nb}."

echo "srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type}" 
srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type} >${test2_output}

################################ WRITE OUT RESULTS #####################################

echo -e " " >> ${RESULT_FILE}
echo -e "TEST CASE II. spatial  model. Gaussian Data. ns = ${ns}, nt = ${nt}, nb = ${nb}, no = ${nb}. " >> ${RESULT_FILE}
echo -e "numRanks numThreadsL1 numThreadsL2 SolverType "  >> ${RESULT_FILE}
echo -e "${num_ranks} ${l1t} ${l2t} ${solver_type} " >> ${RESULT_FILE}
echo -n "est.  mean interpret. param. : "  >> ${RESULT_FILE}
cat ${test2_output} | grep "est.  mean interpret. param." | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "mean interpret. param. original  : 1.3862 -1.60943 1.098612 " >> ${RESULT_FILE}
echo -n "est.  mean parameters        : "  >> ${RESULT_FILE}
cat ${test2_output} | grep "est.  mean parameters" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "mean parameters original         : 1.386294 -5.013283  2.649159 " >> ${RESULT_FILE}
echo -n "estimated mean fixed effects : "  >> ${RESULT_FILE}
cat ${test2_output} | grep "estimated mean fixed effects" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "mean fixed effects original     : -0.294, -0.72 , -0.061" >> ${RESULT_FILE}

################################## TEST III ##########################################
echo " "

ns=492
nss=0
ntFit=50
ntPred=0
nt=$((${ntFit}+${ntPred}))
nb=6
noPerTs=$((2*${ns}))
no=$((2*${ns}*${nt}))

data_type=synthetic
likelihood=gaussian
folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/ns${ns}_nt${nt}_nb${nb}
test3_output=${results_folder}/INLA_testCase_III_ns${ns}_ntFit${nt}_ntPred0_nss${nss}_nb${nb}_${num_ranks}_${l1t}_${l2t}_${solver_type}.txt

echo "TEST CASE III. Spatial-Temporal model. ns = ${ns}, nss = ${nss}, nt = ${nt}, nb = ${nb}, no = ${no}."

echo "srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type}" 
srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type} >${test3_output}

## orig. mean parameters        :  1.38629400 -5.88254100  1.03972100  3.68887900
## orig. mean interpret. param. : 1.38629400 -0.00000023 2.30258418 1.40625832
## orig. mean fixed effects     :  -1, 3, 0.5, 1, -2



################################ WRITE OUT RESULTS #####################################

echo -e " " >> ${RESULT_FILE}
echo -e "TEST CASE III. Spatial-Temporal model. ns = ${ns}, nt = ${nt}, nb = ${nb}, no = ${nb}. " >> ${RESULT_FILE}
echo -e "numRanks numThreadsL1 numThreadsL2 SolverType "  >> ${RESULT_FILE}
echo -e "${num_ranks} ${l1t} ${l2t} ${solver_type} " >> ${RESULT_FILE}
echo -n "est.  mean interpret. param. : "  >> ${RESULT_FILE}
cat ${test3_output} | grep "est.  mean interpret. param." | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "orig. mean interpret. param. : 1.38629400 -0.00000023 2.30258418 1.40625832" >> ${RESULT_FILE}
echo -n "est.  mean parameters        : "  >> ${RESULT_FILE}
cat ${test3_output} | grep "est.  mean parameters" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "orig. mean parameters        :  1.38629400 -5.88254100  1.03972100  3.68887900" >> ${RESULT_FILE}
echo -n "estimated mean fixed effects : "  >> ${RESULT_FILE}
cat ${test3_output} | grep "estimated mean fixed effects" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "orig. mean fixed effects     :  -1, 3, 0.5, 2, 1, -2" >> ${RESULT_FILE}


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
test4_output=${results_folder}/INLA_testCase_IV_ns${ns}_ntFit${nt}_ntPred0_nss${nss}_nb${nb}_${num_ranks}_${l1t}_${l2t}_${solver_type}.txt

echo "TEST CASE IV. Spatial-Temporal model with add. spatial field. ns = ${ns}, nss = ${nss}, nt = ${nt}, nb = ${nb}, no = ${no}."

echo "srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${noPerTs} ${likelihood} ${folder_path} ${solver_type}" 
srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type} >${test4_output}

## interpret.param 1.386 0.405 1.099 1.386 -1.204 1.099
## theta.original  1.386 -4.42 0.634 1.674 -4.608 2.244
## fixed effects   -1 3 0.5 2


################ WRITE OUT RESULTS #################

echo -e " " >> ${RESULT_FILE}
echo -e "TEST CASE IV. Spatial-Temporal model with add. spatial-field. ns = ${ns}, nt = ${nt}, nb = ${nb}, no = ${nb}. " >> ${RESULT_FILE}
echo -e "numRanks numThreadsL1 numThreadsL2 SolverType "  >> ${RESULT_FILE}
echo -e "${num_ranks} ${l1t} ${l2t} ${solver_type} " >> ${RESULT_FILE}
echo -n "est.  mean interpret. param. : "  >> ${RESULT_FILE}
cat ${test4_output} | grep "est.  mean interpret. param." | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "orig. mean interpret. param. : 1.386 0.405 1.099 1.386 -1.204 1.099" >> ${RESULT_FILE}
echo -n "est.  mean parameters        : "  >> ${RESULT_FILE}
cat ${test4_output} | grep "est.  mean parameters" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "orig. mean parameters        :  1.386 -4.42 0.634 1.674 -4.608 2.244" >> ${RESULT_FILE}
echo -n "estimated mean fixed effects : "  >> ${RESULT_FILE}
cat ${test4_output} | grep "estimated mean fixed effects" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "orig. mean fixed effects     :  -1 3 0.5 2" >> ${RESULT_FILE}


########################################################################################################################

##################################
####### POISSON LIKELIHOOD #######
##################################

########################################################################################################################

################################## TEST V ##########################################
echo " "
echo "POISSON TEST CASES"

ns=0
nss=0
ntFit=0
ntPred=0
nt=$((${ntFit}+${ntPred}))
nb=6
no=100

data_type=regression
likelihood=Poisson
folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/${data_type}/${likelihood}Data/nb${nb}_no${no}
test5_output=${results_folder}/INLA_testCase_V_ns${ns}_ntFit${nt}_ntPred0_nss${nss}_nb${nb}_no${no}_${num_ranks}_${l1t}_${l2t}_${likelihood}_${solver_type}.txt

echo "TEST CASE V. regression model. Poisson Data. ns = ${ns}, nss = ${nss}, nt = ${nt}, nb = ${nb}, no = ${no}."

echo "srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type}" 
srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type} >${test5_output}

## fixed effects INLA : 0.166, -1.7075385, -2.0048265692293, 2.09097384070468, 1.79709866484101, -1.03048874777528

################################ WRITE OUT RESULTS #####################################

echo -e " " >> ${RESULT_FILE}
echo -e "TEST CASE V. regression model. Poisson Data. ns = ${ns}, nt = ${nt}, nb = ${nb}, no = ${nb}. " >> ${RESULT_FILE}
echo -e "numRanks numThreadsL1 numThreadsL2 SolverType "  >> ${RESULT_FILE}
echo -e "${num_ranks} ${l1t} ${l2t} ${solver_type} " >> ${RESULT_FILE}
echo -n "No hyperparameters!"  >> ${RESULT_FILE}
cat ${test5_output} | grep "estimated mean fixed effects" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "mean fixed effects INLA     : 0.166, -1.71, -2.00, 2.09, 1.797, -1.03" >> ${RESULT_FILE}




################################## TEST VI ##########################################

ns=483
nss=0
ntFit=0
ntPred=0
nt=$((${ntFit}+${ntPred}))
nb=1
no=966

data_type=synthetic
likelihood=Poisson
folder_path=${base_path}/../data/${data_type}/${likelihood}Data/ns${ns}_nt${nt}_nb${nb}_no${no}
test6_output=${results_folder}/INLA_testCase_VI_ns${ns}_ntFit${nt}_ntPred0_nss${nss}_nb${nb}_no${no}_${num_ranks}_${l1t}_${l2t}_${likelihood}_${solver_type}.txt

echo "TEST CASE VI. spatial model. Poisson Data. ns = ${ns}, nss = ${nss}, nt = ${nt}, nb = ${nb}, no = ${no}."

echo "srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type}"
srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type} >${test6_output}

## fixed effects original: 10

################################ WRITE OUT RESULTS #####################################

echo -e " " >> ${RESULT_FILE}
echo -e "TEST CASE VI. spatial model. Poisson Data. ns = ${ns}, nt = ${nt}, nb = ${nb}, no = ${nb}. " >> ${RESULT_FILE}
echo -e "numRanks numThreadsL1 numThreadsL2 SolverType "  >> ${RESULT_FILE}
echo -e "${num_ranks} ${l1t} ${l2t} ${solver_type} " >> ${RESULT_FILE}
echo -n "est.  mean interpret. param. : "  >> ${RESULT_FILE}
cat ${test6_output} | grep "est.  mean interpret. param." | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "orig. mean interpret. param. : -1.514, 1.093" >> ${RESULT_FILE}
echo -n "est.  mean parameters        : "  >> ${RESULT_FILE}
cat ${test6_output} | grep "est.  mean parameters" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "orig. mean parameters        : -4.9125  2.5536 " >> ${RESULT_FILE}
echo -n "estimated mean fixed effects : "  >> ${RESULT_FILE}
cat ${test6_output} | grep "estimated mean fixed effects" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "mean fixed effects INLA     : 9.834" >> ${RESULT_FILE}


################################## TEST VII ##########################################

ns=363
nss=0
ntFit=30
ntPred=0
nt=$((${ntFit}+${ntPred}))
nb=8
no=16320

data_type=synthetic
likelihood=Poisson
folder_path=${base_path}/../data/${data_type}/${likelihood}Data/ns${ns}_nt${nt}_nb${nb}_no${no}
test7_output=${results_folder}/INLA_testCase_VII_ns${ns}_ntFit${nt}_ntPred0_nss${nss}_nb${nb}_no${no}_${num_ranks}_${l1t}_${l2t}_${likelihood}_${solver_type}.txt

echo "TEST CASE VII. spatial-temporal model. Poisson Data. ns = ${ns}, nss = ${nss}, nt = ${nt}, nb = ${nb}, no = ${no}."

echo "srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type}"
srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type} >${test7_output}

## fixed effects original: 10

################################ WRITE OUT RESULTS #####################################

echo -e " " >> ${RESULT_FILE}
echo -e "TEST CASE VII. spatial-temporal model. Poisson Data. ns = ${ns}, nt = ${nt}, nb = ${nb}, no = ${nb}. " >> ${RESULT_FILE}
echo -e "numRanks numThreadsL1 numThreadsL2 SolverType "  >> ${RESULT_FILE}
echo -e "${num_ranks} ${l1t} ${l2t} ${solver_type} " >> ${RESULT_FILE}
echo -n "est.  mean interpret. param. : "  >> ${RESULT_FILE}
cat ${test7_output} | grep "est.  mean interpret. param." | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "orig. mean interpret. param. : -1.81 1.28 1.12" >> ${RESULT_FILE}
echo -n "est.  mean parameters        : "  >> ${RESULT_FILE}
cat ${test7_output} | grep "est.  mean parameters" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "orig. mean parameters        : -8.74  2.87  6.3" >> ${RESULT_FILE}
echo -n "estimated mean fixed effects : "  >> ${RESULT_FILE}
cat ${test7_output} | grep "estimated mean fixed effects" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "mean fixed effects INLA      : 1.45  1.12  -1.8  -0.23  -1.78  -1.7  1.37  2.01 " >> ${RESULT_FILE}



################################## TEST VIII ##########################################

ns=363
nss=363
ntFit=30
ntPred=0
nt=$((${ntFit}+${ntPred}))
nb=8
no=16320

data_type=synthetic
likelihood=Poisson
folder_path=${base_path}/../data/${data_type}/${likelihood}Data/ns${ns}_nt${nt}_nss${nss}_nb${nb}_no${no}
test8_output=${results_folder}/INLA_testCase_VIII_ns${ns}_ntFit${nt}_ntPred0_nss${nss}_nb${nb}_no${no}_${num_ranks}_${l1t}_${l2t}_${likelihood}_${solver_type}.txt

echo "TEST CASE VIII. spatial-temporal model with add. spatial field. Poisson Data. ns = ${ns}, nss = ${nss}, nt = ${nt}, nb = ${nb}, no = ${no}."

echo "srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type}"
srun -n ${num_ranks} ./call_INLA ${ns} ${ntFit} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type} >${test8_output}

## fixed effects original: 10

################################ WRITE OUT RESULTS #####################################

echo -e " " >> ${RESULT_FILE}
echo -e "TEST CASE VIII. spatial-temporal model with add. spatial field. Poisson Data. ns = ${ns}, nt = ${nt}, nb = ${nb}, no = ${nb}. " >> ${RESULT_FILE}
echo -e "numRanks numThreadsL1 numThreadsL2 SolverType "  >> ${RESULT_FILE}
echo -e "${num_ranks} ${l1t} ${l2t} ${solver_type} " >> ${RESULT_FILE}
echo -n "est.  mean interpret. param. : "  >> ${RESULT_FILE}
cat ${test8_output} | grep "est.  mean interpret. param." | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "orig. mean interpret. param. : -2.25  1.45 0.7 -1.75 0.35" >> ${RESULT_FILE}
echo -n "est.  mean parameters        : "  >> ${RESULT_FILE}
cat ${test8_output} | grep "est.  mean parameters" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "orig. mean parameters        : -9.27  3.29  7.3 -4.4  2.8 " >> ${RESULT_FILE}
echo -n "estimated mean fixed effects : "  >> ${RESULT_FILE}
cat ${test8_output} | grep "estimated mean fixed effects" | cut -d':' -f 2 >> ${RESULT_FILE}
echo -e "mean fixed effects INLA      : 1.38 1.13 -1.8 -0.23 -1.79 -1.7 1.38 2.02 " >> ${RESULT_FILE}