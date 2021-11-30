
export OMP_NESTED=TRUE

# number of threads first layer
l1t=2

# number of threads second layer
l2t=3

#num_threads=$((l1t*l2t-1))

export OMP_NUM_THREADS="${l1t},${l2t}"
echo "OMP_NUM_THREADS=${l1t},${l2t}"

OMP_PLACES=core    #spread    #"{0},{1},{2},...,{5}"
OMP_PROC_BIND=true

export OMP_DISPLAY_ENV=true

echo "./omp_example"
#likwid-pin ./omp_example 
./omp_example

