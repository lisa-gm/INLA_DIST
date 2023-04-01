#mpirun -np 2 ./RGFSolver 42 3 data/A_126_126_ns42_nt3.dat

BASE_PATH=/home/x_gaedkelb
#BASE_PATH=/home/hpc/ihpc/ihpc060h

ns=162
nt=5
nb=6
#no=$((2*${ns}*${nt}))
no=1458

#ns=10242
#nt=30
#nb=6
#no=$((2*${ns}*${nt}))

#ns=10242
#nt=200
#nb=6
#no=$((2*${ns}*${nt}))

data_type=forecasting
#data_type=synthetic

#folder_path=/home/x_gaedkelb/RGF/data/ns${ns}
folder_path=${BASE_PATH}/b_INLA/data/${data_type}/ns${ns}_nt${nt}_nb${nb}
#folder_path=${BASE_PATH}/b_INLA/data/synthetic/fixed_ns${ns}_nt${nt}_nb${nb}
#folder_path=/home/x_gaedkelb/georg/input/ghcn/2019/spatio_temporal/ns4002_nt100

year=2019

solver_type=RGF

#l2t=4
#export OMP_NUM_THREADS="${l2t}"
#echo "OMP_NUM_THREADS=${l2t}"

echo "GPU 0 ./main ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} #>RGF_output.txt"

#CUDA_VISIBLE_DEVICES="0" ./mainEigen ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} #>RGF_output.txt

#srun ./main ${folder_path} ${ns} ${nt} ${nb} ${no} # >${folder_path}/RGF_output.txt
CUDA_VISIBLE_DEVICES="0" ./mainEigen ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} #>RGF_output_s${ns}_nt${nt}_nb${nb}_magmaNative_i2.txt
#mv ${folder_path}/L_factor_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}.dat ${folder_path}/L_factor_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_gpu0.dat
#mv ${folder_path_data}/x_sol_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}.dat ${folder_path_data}/x_sol_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}_gpu0.dat
#mv ${folder_path_data}/log_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}.dat ${folder_path_data}/log_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}_gpu0.dat

#CUDA_VISIBLE_DEVICES="0" nsys profile -o nsys_output_native_noCpyHost.%h.%p ./mainEigen ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >output_mainEigen_ns${ns}_nt${nt}_nb${nb}_native_noCpyHost.txt #

#CUDA_VISIBLE_DEVICES="0" nsys profile -o nsys_output_native_sequential.%h.%p ./mainEigen ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >output_mainEigen_ns${ns}_nt${nt}_nb${nb}_native_sequential.txt #
#CUDA_VISIBLE_DEVICES="0" nsys profile -o nsys_output_native.%h.%p ./mainEigen ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >output_mainEigen_ns${ns}_nt${nt}_nb${nb}_native.txt #
#nsys profile -t nvtx -o nsys_output_%h_%p.txt --stats=true --force-overwrite true ./mainEigen ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type} >output_mainEigen_ns${ns}_nt${nt}_nb${nb}_test.txt

