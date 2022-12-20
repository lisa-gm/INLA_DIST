
MKLROOT?=$(shell echo $$MKLROOT)

BASE_PATH=/home/x_gaedkelb
#BASE_PATH=/home/hpc/ihpc/ihpc060h

############# RGF ################

MKLHOME       = $(MKLROOT)/lib/intel64
BLAHOME       = $(MKLHOME)
LAPHOME       = $(MKLHOME)
SCAHOME       = $(MKLHOME)


RGF_PATH      = ../RGF
include $(RGF_PATH)/make.inc
#############################################################################################################

CXX=ccache mpic++    # kaust
#CXX=mpicxx     # FAU
CXXFLAGS=-O3 -lpthread -lm -lgomp -lgfortran -fPIC -fopenmp -fPIC -lm #-Wall -llikwid -DLIKWID_PERFMON

DEBUG= #-g #-Wall

# LIBRARIES
LIBPARDISO=-lpardiso720-GNU840-X86-64-RINLA_newSelInv		#version with deterministic determinant! & fixed selected inversion
							#expects the path in .bashrc (LD_LIBRARY_PATH, LIBRARY)
LIBMKL=-L$(MKLROOT)/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core
#LIBRGF=-lrgf


#LIBCHOLMOD=-lcholmod
ARMADILLO=-larmadillo

LIBS=$(LIBPARDISO) $(LIBMKL) $(ARMADILLO) #$(LIBCHOLMOD)

# INCLUDES
INCEIGEN=-I/usr/include/eigen3
#INCEIGEN=-I$(BASE_PATH)/applications/eigen-3.3.9
INCBFGS=-I$(BASE_PATH)/applications/adapted_LBFGSpp/include
#INCCHOLMOD= -I/usr/include/suitesparse

INCLUDES=$(INCEIGEN) $(INCBFGS) #$(INCCHOLMOD)

RGF_INCLUDEDIR  = $(INCCUD) $(INCMAG)
RGF_LIBS	    = $(LAPACK) $(OPENMP) $(CUDA) $(MAGMA)

RGF_CC_FILES   = $(RGF_PATH)/Utilities.o #RGF/main.o
RGF_CU_FILES   = $(RGF_PATH)/CWC_utility.o

$(info ============== Library paths ==============)
$(info MKL      : $(MKLHOME))
$(info BFGS     : $(INCBFGS))
$(info CUDA     : $(CUDAHOME))
$(info MAGMA    : $(MAGMA))
$(info EIGEN    : $(INCEIGEN))
$(info RGF      : $(RGF_PATH))


$(info )
$(info ============== Compiling ==============)


all: call_INLA

#librgf.a: $(CC_FILES) $(CU_FILES)
#	ar rvs $@ $^

#librgf.so: $(CC_FILES) $(CU_FILES)
#	$(LOADER) --shared $(CC_FILES) $(CU_FILES) $(LOADFLAGS) $(LIBS) -lm -o $@ $(DEBUG) -fPIC

#RGF/main: $(RGF_CC_FILES) $(RGF_CU_FILES) 
#	$(RGF_LOADER) $^ $(RGF_LOADFLAGS) $(RGF_LIBS) $(OPENMP) -lm -o $@ $(DEBUG)

#RGF/main.o: RGF/main.C
#	$(CXX) -c $^ $(RGF_CXXFLAGS) $(DEBUG) $(RGF_FLAGS) $(RGF_INCLUDEDIR) -fPIC -o $@	

$(RGF_PATH)/Utilities.o: $(RGF_PATH)/Utilities.C
	$(RGF_MPICXX) -c $^ $(RGF_CXXFLAGS) $(DEBUG) $(RGF_FLAGS) $(RGF_INCLUDEDIR) -fPIC -o $@

$(RGF_PATH)/CWC_utility.o: $(RGF_PATH)/CWC_utility.cu
	$(NVCC) -c $^ $(NVCCFLAGS) $(INCCUD) --compiler-options '-fPIC' -o $@

call_INLA: call_INLA.o PostTheta.o PardisoSolver.o RGFSolver.o $(RGF_CC_FILES) $(RGF_CU_FILES) #solver_cholmod.o 
	$(CXX) $^ $(LIBS) $(RGF_LIBS) $(CXXFLAGS) $(DEBUG) -lm -o $@  #$(LDFLAGS) $(LDLIBS) 

call_INLA.o: call_INLA.cpp #call_INLA.cpp
	$(CXX) -c $< $(CXXFLAGS) $(RGF_CXXFLAGS) $(RGF_FLAGS) $(INCLUDES) $(RGF_INCLUDEDIR) -o $@ 

%.o : %.cpp %.h
	$(CXX) -c $< $(CXXFLAGS) $(INCLUDES) $(RGF_INCLUDEDIR) -o $@ 

#RGF/librgf.a:
#	$(MAKE) -C RGF

#RGF/librgf.so:
#	$(MAKE) -C RGF


clean:	
	rm -f *.o call_INLA
	rm -f RGF/*.o
	rm -f RGF/main

#rm -f librgf.so
#rm -f librgf.a



