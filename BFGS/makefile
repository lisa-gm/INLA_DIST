

#CXX_COMPILER=g++
#CXX_FLAGS= -lm -lcholmod -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -lgomp -lgfortran -fPIC  -fopenmp -fPIC

#SRC_FILES= example.cpp theta_function.cpp ../read_write_functions.cpp

#INCLUDES= -I/usr/include/eigen3 -I/home/x_gaedkelb/applications/LBFGSpp/include -I/usr/include/suitesparse

MKLROOT?=$(shell echo $$MKLROOT)
#PARDISO_LIB=/home/x_gaedkelb/applications/pardiso/libpardiso700-GNU840-X86-64-RINLA.so


#example: $(SRC_FILES)
#	$(CXX_COMPILER) $(SRC_FILES) -o example $(INCLUDES) $(MKL_LIB) $(PARDISO_LIB) $(CXX_FLAGS)

#example: example.cpp
#	rm -f example || true
#	g++ $< -o $@ -I/usr/include/eigen3 -lm -I/home/x_gaedkelb/applications/LBFGSpp/include -I/usr/include/suitesparse -lcholmod -lm \
	-fopenmp \
	/home/x_gaedkelb/applications/pardiso/libpardiso700-GNU840-X86-64-RINLA.so -L/opt/intel/oneapi/mkl/latest/lib/intel64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -lgomp -lgfortran -fPIC  -fopenmp -fPIC -lm


example: example.cpp
	rm -f example || true
	g++ $< -o $@ -I/usr/include/eigen3 -lm -I/home/x_gaedkelb/applications/LBFGSpp/include -I/usr/include/suitesparse -lcholmod -lm \
	-fopenmp \
	/home/x_gaedkelb/applications/pardiso/libpardiso700-GNU840-X86-64-RINLA.so -L${MKLROOT}/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -lgomp -lgfortran -fPIC  -fopenmp -fPIC -lm

#example: example.cpp
#	rm -f test_file || true
#	g++ $< -o $@ -I/usr/include/eigen3 -lm -I/home/x_gaedkelb/applications/LBFGSpp/include -I/usr/include/suitesparse -lcholmod -lm \
	-fopenmp \
	/home/x_gaedkelb/applications/pardiso/libpardiso700-GNU840-X86-64-RINLA.so -lblas -llapack -lpthread -lm -lgomp -lgfortran -fPIC  -fopenmp -fPIC -lm


# -std=c++17
#g++ -I/path/to/eigen -I/path/to/lbfgspp/include -O2 example.cpp

test_file: test_file.cpp
	g++ $< -o $@ -fopenmp -I/usr/include/eigen3 -lm -I/home/x_gaedkelb/applications/LBFGSpp/include -I/usr/include/suitesparse -lcholmod -lm \
	/home/x_gaedkelb/applications/pardiso/libpardiso700-GNU840-X86-64-RINLA.so -L/opt/intel/oneapi/mkl/latest/lib/intel64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -lgomp -lgfortran -fPIC  -fopenmp -fPIC -lm

test_omp: test_omp.cpp
	g++ $< -o $@ -fopenmp

clean: 
	rm -f example
	rm -f test_file
