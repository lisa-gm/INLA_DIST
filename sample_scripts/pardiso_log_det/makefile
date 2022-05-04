


INCEIGEN=-I/usr/include/eigen3
#LIBPARDISO=/home/x_gaedkelb/applications/pardiso/libpardiso720-GNU840-X86-64-RINLA_new.so
LIBPARDISO=/home/x_gaedkelb/applications/pardiso/libpardiso700-GNU840-X86-64-RINLA.so


main: main.cpp PardisoSolver.cpp
	g++ $^ -o $@ $(INCEIGEN) -fopenmp $(LIBPARDISO) \
	-L${MKLROOT}/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -lgomp -lgfortran -fPIC  -fopenmp -fPIC -lm
