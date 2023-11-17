
MKLROOT?=$(shell echo $$MKLROOT)

# PARDISO only -- doesn't require CUDA or MAGMA
#############################################################################################################
### BASE_PATH TO BE ADAPTED
#BASE_PATH=/home/x_gaedkelb
#BASE_PATH=/users/lgaedkem
BASE_PATH=/home/hpc/ihpc/ihpc060h
#############################################################################################################

#CXX=CC     #mpic++
CXX=mpicxx
CXXFLAGS=-O3 -lpthread -lm -lgomp -lgfortran -fPIC -fopenmp -fPIC -lm -Wall #-llikwid -DLIKWID_PERFMON

# LIBRARIES
LIBPARDISO=-lpardiso720-GNU840-X86-64-RINLA_newSelInv 	#expects the path in .bashrc (LD_LIBRARY_PATH, LIBRARY)
LIBMKL=-L${MKLROOT}/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core

ARMADILLO=-larmadillo

LIBS=$(LIBPARDISO) $(LIBMKL) $(ARMADILLO)

# INCLUDES
#INCEIGEN=-I/usr/include/eigen3
INCEIGEN=-I$(BASE_PATH)/applications/eigen

INCBFGS=-I$(BASE_PATH)/applications/adapted_LBFGSpp/include
#INCCHOLMOD= -I/usr/include/suitesparse

INCLUDES=$(INCEIGEN) $(INCBFGS) #$(INCCHOLMOD)

$(info =============== Library paths ==============)
$(info MKL      : $(MKLROOT))
$(info EIGEN    : $(INCEIGEN))

$(info )
$(info ===== Compiling w/ BTA dummy version =======)

all: call_INLA

call_INLA: call_INLA.o PostTheta.o PardisoSolver.o RGFSolver_dummy.o
	$(CXX) $^ $(LIBS) $(CXXFLAGS) $(DEBUG) -lm -o $@  #$(LDFLAGS) $(LDLIBS) 

call_INLA.o: call_INLA.cpp
	$(CXX) -c $< $(CXXFLAGS) $(INCLUDES) -o $@ 

%.o : %.cpp %.h
	$(CXX) -c $< $(CXXFLAGS) $(INCLUDES) -o $@ 

clean:	
	rm -f *.o call_INLA
