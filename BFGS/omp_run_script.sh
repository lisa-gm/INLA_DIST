
export OMP_NESTED=TRUE

# number of threads first layer
l1t=4

# number of threads second layer
l2t=3

export OMP_NUM_THREADS="${l1t},${l2t}"
echo "OMP_NUM_THREADS=${l1t},${l2t}"

./test_omp 
echo "./test_omp"

