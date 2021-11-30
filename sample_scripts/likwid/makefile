



mpi_example: mpi_example.cpp
	mpic++ $< -o $@ -fopenmp


perfctr_example: perfctr_example.cpp
	g++ -DLIKWID_PERFMON $< -o $@ -llikwid


omp_example: test_omp.cpp
	g++ $< -o $@ -fopenmp
