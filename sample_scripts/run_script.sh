
ns=492
nt=16
nb=2
no=7872

#ns=42
#nt=3
#nb=2
#no=35328
	
export PARDISOLICMESSAGE=1
export OMP_NUM_THREADS=8

#folder_path=/home/x_gaedkelb/RGF/data/ns${ns}_nt${nt}
folder_path=/home/x_gaedkelb/b_INLA/data/synthetic/ns${ns}_nt${nt}

echo "./pardiso_reuse_symf_ex2 ${ns} ${nt} ${nb} ${no} ${folder_path}" 
./pardiso_reuse_symf_ex2 ${ns} ${nt} ${nb} ${no} ${folder_path}