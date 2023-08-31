MKLROOT?=$(shell echo $$MKLROOT)

#BASE_PATH=/home/x_gaedkelb
BASE_PATH=/home/hpc/ihpc/ihpc060h
#BASE_PATH=/users/lgaedkem

# export cholmod paths
#source ~/env/cholmod.sh

MKLHOME       = $(MKLROOT)/lib/intel64
#############################################################################################################

#CXX=CC    # cscs
CXX=mpicxx     # FAU
CXXFLAGS=-O3 -lpthread -lm -lgomp -lgfortran -fPIC -fopenmp -fPIC -lm #-Wall -llikwid -DLIKWID_PERFMON

# LIBRARIES
LIBMKL=-L$(MKLROOT)/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core
LIBCHOLMOD=-L$(BASE_PATH)/applications/SuiteSparse/CHOLMOD/build

LIBCHOLMOD=-lcholmod
#ARMADILLO=-larmadillo
# $(ARMADILLO)
LIBS=$(LIBMKL) $(LIBCHOLMOD)

# INCLUDES
#INCEIGEN=-I/usr/include/eigen3
INCEIGEN=-I$(BASE_PATH)/applications/eigen
INCCHOLMOD= -I$(BASE_PATH)/applications/SuiteSparse/CHOLMOD/Include
INCSUITESPARSE= -I$(BASE_PATH)/applications/SuiteSparse/SuiteSparse_config

INCLUDES=$(INCEIGEN) $(INCCHOLMOD) $(INCSUITESPARSE)

$(info ============== Library paths ==============)
$(info MKL      : $(MKLHOME))
$(info EIGEN    : $(INCEIGEN))


$(info )
$(info ============== Compiling ==============)


all: main

main: main.o EigenCholSolver.o #solver_cholmod.o 
	$(CXX) $^ $(LIBS) $(RGF_LIBS) $(CXXFLAGS) $(DEBUG) -lm -o $@  #$(LDFLAGS) $(LDLIBS) 

main.o: main.C
	$(CXX) -c $< $(CXXFLAGS) $(INCLUDES) -o $@ 

%.o : %.cpp %.h
	$(CXX) -c $< $(CXXFLAGS) $(INCLUDES) -o $@ 



clean:	
	rm -f *.o main




