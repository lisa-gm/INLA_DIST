
example: example.cpp
	rm -f example || true
	g++ $< -o $@ -I/usr/include/eigen3 -lm -I/home/x_gaedkelb/applications/LBFGSpp/include -I/usr/include/suitesparse -lcholmod -lm \
	-fopenmp \
	/home/x_gaedkelb/applications/pardiso/libpardiso700-GNU840-X86-64-RINLA.so -L/opt/intel/oneapi/mkl/latest/lib/intel64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -lgomp -lgfortran -fPIC  -fopenmp -fPIC -lm


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
