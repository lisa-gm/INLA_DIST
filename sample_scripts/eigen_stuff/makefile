
MKLHOME       = $(MKLROOT)/lib/intel64
LIBMKL=-L$(MKLROOT)/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core

CXX_FLAGS=-lpthread -lm -lgomp -lgfortran -fPIC  -fopenmp -fPIC -lm

INC_EIGEN=-I/usr/include/eigen3




all: main

# -g -fsanitize=address
main: main.C 
	g++ $^ -o $@ -g -fsanitize=address $(INC_EIGEN) $(LIB_MKL) $(CXX_FLAGS)
