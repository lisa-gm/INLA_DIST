BASE_PATH=/home/x_gaedkelb
#BASE_PATH=/home/hpc/ihpc/ihpc060h

############# RGF ################

MKLHOME       = $(MKLROOT)/lib/intel64
BLAHOME       = $(MKLHOME)
LAPHOME       = $(MKLHOME)
SCAHOME       = $(MKLHOME)

#RGF_PATH      = ../../RGF
RGF_PATH      = /home/x_gaedkelb/b_INLA/RGF
include $(RGF_PATH)/make.inc
#############################################################################################################

CXX=mpic++
CXXFLAGS= -g -lpthread -lm -lgomp -lgfortran -fPIC -fopenmp -fPIC -lm -O0 # -O3 #-Wall -llikwid -DLIKWID_PERFMON

DEBUG= -g #-Wall


INCEIGEN=-I/usr/include/eigen3
#INCEIGEN=-I${BASE_PATH}/applications/eigen

#LIBCHOLMOD=-lcholmod
ARMADILLO=-larmadillo

LIBS=$(LIBPARDISO) $(LIBMKL) $(ARMADILLO) #$(LIBCHOLMOD)

LIBPARDISO=${BASE_PATH}/applications/pardiso/libpardiso720-GNU840-X86-64-RINLA_newSelInv.so
#LIBPARDISO=/home/x_gaedkelb/applications/pardiso/libpardiso700-GNU840-X86-64-RINLA.so

### RGF stuff
RGF_INCLUDEDIR  = $(INCCUD) $(INCMAG)
RGF_LIBS	    = $(LAPACK) $(OPENMP) $(CUDA) $(MAGMA)

RGF_OWN_INCLUDE= -I$(RGF_PATH)
RGF_CC_FILES   = $(RGF_PATH)/Utilities.o #RGF/main.o
RGF_CU_FILES   = $(RGF_PATH)/CWC_utility.o

$(info ============== Library paths ==============)
$(info MKL      : $(MKLHOME))
$(info CUDA     : $(CUDAHOME))
$(info MAGMA    : $(MAGMA))
$(info EIGEN    : $(INCEIGEN))
$(info RGF      : $(RGF_PATH))

$(info )
$(info ============== Compiling ==============)

all: main

#mainEigenMPI: mainEigenMPI.cpp PardisoSolver.cpp
#	mpicxx $^ -o $@ $(INCEIGEN) -fopenmp $(LIBPARDISO) \
	-L${MKLROOT}/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -lgomp -lgfortran -fPIC  -fopenmp -fPIC -lm

$(RGF_PATH)/Utilities.o: $(RGF_PATH)/Utilities.C
	$(RGF_MPICXX) -c $^ $(RGF_CXXFLAGS) $(DEBUG) $(RGF_FLAGS) $(RGF_INCLUDEDIR) -fPIC -o $@

$(RGF_PATH)/CWC_utility.o: $(RGF_PATH)/CWC_utility.cu
	$(NVCC) -c $^ $(NVCCFLAGS) $(INCCUD) --compiler-options '-fPIC' -o $@

main: main.o PardisoSolver.o RGFSolver.o $(RGF_CC_FILES) $(RGF_CU_FILES) #solver_cholmod.o 
	$(CXX) $^ $(LIBS) $(RGF_LIBS) $(CXXFLAGS) $(DEBUG) -lm -o $@  #$(LDFLAGS) $(LDLIBS) 

main.o: main.cpp #call_INLA.cpp
	$(CXX) -c $< $(CXXFLAGS) $(RGF_CXXFLAGS) $(RGF_FLAGS) $(INCEIGEN) $(RGF_INCLUDEDIR) -o $@ 

%.o : %.cpp %.h
	$(CXX) -c $< $(CXXFLAGS) $(INCEIGEN) $(RGF_INCLUDEDIR) -o $@ 

#main: main.cpp PardisoSolver.cpp RGFSolver.cpp
#	$(CXX) $^ -o $@ $(INCEIGEN) -fopenmp $(LIBPARDISO) \
	-L${MKLROOT}/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -lgomp -lgfortran -fPIC  -fopenmp -fPIC -lm

clean:	
	rm -f *.o main
	rm -f RGF/*.o
	rm -f RGF/main
