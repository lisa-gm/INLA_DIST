#mpirun -np 2 ./RGFSolver 42 3 data/A_126_126_ns42_nt3.dat


ns=$1
nt=$2
no=$3

#folder_path=/home/gaedkem/spat_temp_m_pardiso/matrices_theta/theta_noise_5/theta_5_-10_2.5_1_ns${ns}_nt${nt}
#folder_path_data=/home/gaedkem/spat_temp_m_pardiso/data/temperature_data_2019/ns${ns}_nt${nt}

folder_path=/home/x_gaedkelb/spat_temp_m_pardiso/matrices_theta/theta_noise_5/theta_5_-10_2.5_1_ns${ns}_nt${nt}
folder_path_data=/home/x_gaedkelb/spat_temp_m_pardiso/data/temperature_data_2019/ns${ns}_nt${nt}

nb=2
year=2019

echo "GV100"
echo "GPU 0 ./RGFSolver ${folder_path} ${ns} ${nt} ${nb} ${folder_path_data} ${no} >${folder_path_data}/RGF_output_sel_inv.txt"

CUDA_VISIBLE_DEVICES="0" ./RGFSolver ${folder_path} ${ns} ${nt} ${nb} ${folder_path_data} ${no} >${folder_path_data}/RGF_output_sel_inv_gpu0.txt
#mv ${folder_path}/L_factor_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}.dat ${folder_path}/L_factor_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_gpu0.dat
mv ${folder_path_data}/x_sol_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}.dat ${folder_path_data}/x_sol_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}_gpu0.dat
mv ${folder_path_data}/log_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}.dat ${folder_path_data}/log_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_year${year}_gpu0.dat

#echo "RTX 4000"
#echo "GPU 1 ./RGFSolver ${folder_path} ${ns} ${nt} ${nb} ${folder_path_data} ${no} >${folder_path}/RGF_output_sel_inv.txt"

#CUDA_VISIBLE_DEVICES="1" ./RGFSolver ${folder_path} ${ns} ${nt} ${nb} ${folder_path_data} ${no} >${folder_path}/RGF_output_sel_inv_gpu1.txt
#mv ${folder_path}/L_factor_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}.dat ${folder_path}/L_factor_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_gpu1.dat
#mv ${folder_path}/x_sol_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}.dat ${folder_path}/x_sol_RGF_ns${ns}_nt${nt}_nb${nb}_no${no}_gpu1.dat


#./RGFSolver ${folder_path} ${ns} ${nt} ${nb} ${folder_path_data} ${no} >${folder_path}/RGF_output_sel_inv.txt
