#mpirun -np 2 ./RGFSolver 42 3 data/A_126_126_ns42_nt3.dat


ns=$1
nt=$2
no=$3

#folder_path=/home/x_gaedkelb/RGF/data/ns${ns}
folder_path=/home/x_gaedkelb/RGF/data/ns${ns}_nt${nt}

nb=2
year=2019

echo "GV100"
echo "GPU 0 ./mainConstInd ${folder_path} ${ns} ${nt} ${nb} ${no} >${folder_path}/RGF_output_sel_inv.txt"

CUDA_VISIBLE_DEVICES="0" ./mainConstInd ${folder_path} ${ns} ${nt} ${nb} ${folder_path_data} ${no} >${folder_path}/RGF_output.txt
#mv ${folder_path}/L_factor_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}.dat ${folder_path}/L_factor_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_gpu0.dat
#mv ${folder_path_data}/x_sol_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}.dat ${folder_path_data}/x_sol_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}_gpu0.dat
#mv ${folder_path_data}/log_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}.dat ${folder_path_data}/log_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}_gpu0.dat

