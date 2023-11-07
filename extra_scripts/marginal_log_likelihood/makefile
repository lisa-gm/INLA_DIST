
MKLROOT?=$(shell echo $$MKLROOT)

#BASE_PATH=/home/x_gaedkelb
BASE_PATH=/home/hpc/ihpc/ihpc060h

############# RGF ################

MKLHOME       = $(MKLROOT)/lib/intel64
BLAHOME       = $(MKLHOME)
LAPHOME       = $(MKLHOME)
SCAHOME       = $(MKLHOME)

#############################################################################################################

#CXX=ccache mpic++    # kaust
CXX=mpicxx     # FAU
CXXFLAGS=-O3 -lpthread -lm -lgomp -lgfortran -fPIC -fopenmp -fPIC -lm #-Wall -llikwid -DLIKWID_PERFMON

DEBUG=  #-g #-Wall -lnvToolsExt

# LIBRARIES
LIBPARDISO=-lpardiso720-GNU840-X86-64-RINLA_newSelInv		#version with deterministic determinant! & fixed selected inversion
							#expects the path in .bashrc (LD_LIBRARY_PATH, LIBRARY)
LIBMKL=-L$(MKLROOT)/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core
#LIBRGF=-lrgf


#LIBCHOLMOD=-lcholmod
ARMADILLO=-larmadillo

LIBS=$(LIBPARDISO) $(LIBMKL) $(ARMADILLO) #$(LIBCHOLMOD)

# INCLUDES
#INCEIGEN=-I/usr/include/eigen3
INCEIGEN=-I$(BASE_PATH)/applications/eigen
#INCEIGEN=-I$(BASE_PATH)/applications/eigen-3.3.9
INCBFGS=-I$(BASE_PATH)/applications/adapted_LBFGSpp/include
#INCCHOLMOD= -I/usr/include/suitesparse

INCLUDES=$(INCEIGEN) $(INCBFGS) #$(INCCHOLMOD)


$(info ============== Library paths ==============)
$(info MKL      : $(MKLHOME))
$(info BFGS     : $(INCBFGS))
$(info EIGEN    : $(INCEIGEN))


$(info )
$(info ============== Compiling ==============)


all: main



main: main.o #solver_cholmod.o 
	$(CXX) $^ $(LIBS) $(CXXFLAGS) $(DEBUG) -lm -o $@  #$(LDFLAGS) $(LDLIBS) 

main.o: main.C #call_INLA.cpp
	$(CXX) -c $< $(CXXFLAGS) $(INCLUDES) -o $@ 


clean:	
	rm -f *.o main






