## mpi run script


num_ranks=$1

likwid-mpirun -np ${num_ranks} ./mpi_example