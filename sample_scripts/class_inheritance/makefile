
#INC_EIGEN=-I/usr/include/eigen3

#all: main

#main: main.cpp Solver.o RGFSolver.o PardisoSolver.o
#	g++ $^ -o $@ $(INC_EIGEN)

#%.o: %.h
#	g++ -c $< -o $@ $(INC_EIGEN)

#clean:
#	rm *.o main


main: main.cpp
	g++ main.cpp -o main \
	-I/usr/include/eigen3 -lm \
	-fopenmp \
	/home/x_gaedkelb/applications/pardiso/libpardiso700-GNU840-X86-64-RINLA.so \
	-L/opt/intel/oneapi/mkl/latest/lib/intel64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core \
	-lpthread -lm -lgomp -lgfortran -fPIC  -fopenmp -fPIC -lm




