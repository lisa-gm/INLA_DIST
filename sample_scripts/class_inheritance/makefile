
CXX_FLAGS=-lpthread -lm -lgomp -lgfortran -fPIC  -fopenmp -fPIC -lm

INC_EIGEN=-I/usr/include/eigen3

LIB_MKL=-L/opt/intel/oneapi/mkl/latest/lib/intel64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core
LIB_PARDISO=-lpardiso700-GNU840-X86-64-RINLA




all: main

# -g -fsanitize=address
main: main.cpp RGFSolver.o PardisoSolver.o
	g++ $^ -o $@ -g -fsanitize=address $(INC_EIGEN) $(LIB_PARDISO) $(LIB_MKL) $(CXX_FLAGS)

# -g -fsanitize=address
%.o: %.cpp %.h
	g++ -c $< -o $@ -g -fsanitize=address $(INC_EIGEN) $(LIB_PARDISO) $(LIB_MKL) $(CXX_FLAGS)

clean:
	rm *.o main


#main: main.cpp
#	g++ main.cpp -o main \
	-g -fsanitize=address \
	-I/usr/include/eigen3 -lm \
	-fopenmp \
	/home/x_gaedkelb/applications/pardiso/libpardiso700-GNU840-X86-64-RINLA.so \
	-L/opt/intel/oneapi/mkl/latest/lib/intel64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core \
	-lpthread -lm -lgomp -lgfortran -fPIC  -fopenmp -fPIC -lm

#clean:
#	rm main


