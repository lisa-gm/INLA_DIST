# hard code example to see if code runs

#ns=2252
#nb=2
#no=11646

ns=$1
nt=$2
nb=$3
no=$4


export PARDISOLICMESSAGE=1

export OMP_NESTED=TRUE
omp_threads="8,4"


export OMP_NUM_THREADS=${omp_threads}
echo "OMP_NUM_THREADS = ${omp_threads}"

folder_path=/home/x_gaedkelb/b_INLA/data/ns${ns}_nt${nt}

echo "./example ${ns} ${nt} ${nb} ${no} ${folder_path}"
./example ${ns} ${nt} ${nb} ${no} ${folder_path}
