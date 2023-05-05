#mpirun -np 2 ./RGFSolver 42 3 data/A_126_126_ns42_nt3.dat


#ns=162
#nt=5
#nb=6
#no=$((2*${ns}*${nt}))
#no=1458

#ns=492
#nt=100
#nb=6
#no=$((2*${ns}*${nt}))

ns=10242
#ns=20252
nt=30
nb=6
no=$((2*${ns}*${nt}))

#data_type=forecasting
data_type=synthetic

#folder_path=/home/x_gaedkelb/RGF/data/ns${ns}
folder_path=/home/hpc/ihpc/ihpc060h/b_INLA/data/synthetic/ns${ns}_nt${nt}
#folder_path=/home/x_gaedkelb/b_INLA/data/synthetic/ns${ns}_nt${nt}
#folder_path=/home/x_gaedkelb/georg/input/ghcn/2019/spatio_temporal/ns4002_nt100

nb=2
year=2019

solver_type=RGF

l2t=2
export OMP_NUM_THREADS="${l2t}"
echo "OMP_NUM_THREADS=${l2t}"

echo "GPU 0 ./mainEigen ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} #>RGF_output.txt"
#CUDA_VISIBLE_DEVICES="0" ./mainEigen ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} #>RGF_output.txt

#srun ./mainEigen ${folder_path} ${ns} ${nt} ${nb} ${no} # >${folder_path}/RGF_output.txt
CUDA_VISIBLE_DEVICES="0" ./mainEigen ${folder_path} ${ns} ${nt} ${nb} ${no} # >${folder_path}/RGF_output.txt
#mv ${folder_path}/L_factor_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}.dat ${folder_path}/L_factor_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_gpu0.dat
#mv ${folder_path_data}/x_sol_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}.dat ${folder_path_data}/x_sol_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}_gpu0.dat
#mv ${folder_path_data}/log_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}.dat ${folder_path_data}/log_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}_gpu0.dat

