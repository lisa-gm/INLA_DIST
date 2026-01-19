
MKLROOT?=$(shell echo $$MKLROOT)

BASE_PATH=$(HOME)

############# BTA ################

MKLHOME       = $(MKLROOT)/lib/intel64
BLAHOME       = $(MKLHOME)
LAPHOME       = $(MKLHOME)
SCAHOME       = $(MKLHOME)


BTA_PATH      = ../BTA
include $(BTA_PATH)/make.inc

# INCLUDES -- TODO: adapt !!
INCEIGEN=-I$(BASE_PATH)/applications/eigen
INCBFGS=-I$(BASE_PATH)/applications/adapted_LBFGSpp/include
#INCCHOLMOD= -I/usr/include/suitesparse
#INCCHOLMOD= -I$(BASE_PATH)/applications/SuiteSparse/CHOLMOD/Include
#INCSUITESPARSE= -I$(BASE_PATH)/applications/SuiteSparse/SuiteSparse_config
#############################################################################################################

CXX=mpicxx    # cscs
#CXX=mpicxx     # FAU
#CXXFLAGS=-O3 -lpthread -lm -lgomp -lgfortran -fPIC -fopenmp -fPIC -lm -march=native -ffast-math -funroll-loops -std=c++14 #  -Wall -llikwid -DLIKWID_PERFMON
CXXFLAGS=-O3 -lpthread -lm -lgomp -lgfortran -fPIC -fopenmp -lopenblas -std=c++14 -fPIC -lm -march=native -ffast-math -funroll-loops -ftree-vectorize -DMPICH_IGNORE_CXX_SEEK #-Wall -llikwid -DLIKWID_PERFMON

DEBUG=  -lnvToolsExt # -g -Wall

# LIBRARIES
# LIBPARDISO=-lpardiso720-GNU840-X86-64-RINLA_newSelInv 	#expects the path in .bashrc (LD_LIBRARY_PATH, LIBRARY)
# LIBMKL=-L$(MKLROOT)/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core


# LIBCHOLMOD=-lcholmod
# ARMADILLO=-larmadillo

LIBS=$(LIBPARDISO) $(LIBMKL) #$(ARMADILLO) #$(LIBCHOLMOD)

INCLUDES=$(INCEIGEN) $(INCBFGS) #$(INCCHOLMOD) $(INCSUITESPARSE)

BTA_INCLUDEDIR  = $(INCCUD) $(INCMAG)
BTA_LIBS	    = $(OPENBLAS) $(OPENMP) $(CUDA) $(MAGMA)

BTA_CC_FILES   = $(BTA_PATH)/Utilities.o #BTA/main.o
BTA_CU_FILES   = $(BTA_PATH)/CWC_utility.o

$(info ============== Library paths ==============)
#$(info MKL      : $(MKLHOME))
$(info OPENBLAS : $(OPENBLAS))
$(info CUDA     : $(CUDAHOME))
$(info MAGMA    : $(MAGMA))
$(info BFGS     : $(INCBFGS))
$(info EIGEN    : $(INCEIGEN))
$(info BTA      : $(BTA_PATH))


$(info )
$(info ============== Compiling ==============)


all: call_INLA_sliding_windows #call_INLA

#BTA/main: $(BTA_CC_FILES) $(BTA_CU_FILES) 
#	$(BTA_LOADER) $^ $(BTA_LOADFLAGS) $(BTA_LIBS) $(OPENMP) -lm -o $@ $(DEBUG)

#BTA/main.o: BTA/main.C
#	$(CXX) -c $^ $(BTA_CXXFLAGS) $(DEBUG) $(BTA_FLAGS) $(BTA_INCLUDEDIR) -fPIC -o $@	

$(BTA_PATH)/Utilities.o: $(BTA_PATH)/Utilities.C
	$(BTA_MPICXX) -c $^ $(BTA_CXXFLAGS) $(DEBUG) $(BTA_FLAGS) $(BTA_INCLUDEDIR) -fPIC -o $@

$(BTA_PATH)/CWC_utility.o: $(BTA_PATH)/CWC_utility.cu
	$(NVCC) -c $^ $(NVCCFLAGS) $(BTA_CXXFLAGS) $(INCCUD) --compiler-options '-fPIC' -o $@

call_INLA_sliding_windows: call_INLA_sliding_windows.o PostTheta.o PardisoSolver_dummy.o BTASolver.o EigenCholSolver.o $(BTA_CC_FILES) $(BTA_CU_FILES) #solver_cholmod.o 
	$(CXX) $^ $(LIBS) $(BTA_LIBS) $(CXXFLAGS) $(DEBUG) -lm -o $@  #$(LDFLAGS) $(LDLIBS) 

call_INLA_sliding_windows.o: call_INLA_sliding_windows.cpp
	$(CXX) -c $< $(CXXFLAGS) $(BTA_CXXFLAGS) $(BTA_FLAGS) $(INCLUDES) $(BTA_INCLUDEDIR) -o $@ 

%.o : %.cpp %.h
	$(CXX) -c $< $(CXXFLAGS) $(INCLUDES) $(BTA_INCLUDEDIR) -o $@ 


clean:	
	rm -f *.o call_INLA_sliding_windows call_INLA
	rm -f BTA/*.o
	rm -f BTA/main





