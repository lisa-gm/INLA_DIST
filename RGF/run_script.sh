#mpirun -np 2 ./RGFSolver 42 3 data/A_126_126_ns42_nt3.dat


#ns=$1
#nt=$2
#no=$3

ns=492
nt=16
no=7872

#ns=4002
#nt=100
#no=1188566

#folder_path=/home/x_gaedkelb/RGF/data/ns${ns}
folder_path=/home/x_gaedkelb/b_INLA/data/synthetic/ns${ns}_nt${nt}
#folder_path=/home/x_gaedkelb/georg/input/ghcn/2019/spatio_temporal/ns4002_nt100

nb=2
year=2019

solver_type=RGF

echo "GV100"
echo "GPU 0 ./main ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >RGF_output.txt"

#CUDA_VISIBLE_DEVICES="0" ./mainEigen ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} #>RGF_output.txt

CUDA_VISIBLE_DEVICES="0" ./main ${folder_path} ${ns} ${nt} ${nb} ${no} # >${folder_path}/RGF_output.txt
#mv ${folder_path}/L_factor_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}.dat ${folder_path}/L_factor_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_gpu0.dat
#mv ${folder_path_data}/x_sol_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}.dat ${folder_path_data}/x_sol_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}_gpu0.dat
#mv ${folder_path_data}/log_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}.dat ${folder_path_data}/log_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}_gpu0.dat

