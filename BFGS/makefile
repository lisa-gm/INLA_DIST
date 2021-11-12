
MKLROOT?=$(shell echo $$MKLROOT)

############# RGF ################

MKLHOME       = $(MKLROOT)/lib/intel64
BLAHOME       = $(MKLHOME)
LAPHOME       = $(MKLHOME)
SCAHOME       = $(MKLHOME)

include ./RGF/make.inc
#############################################################################################################

CXX=g++
CXXFLAGS=-lpthread -lm -lgomp -lgfortran -fPIC -fopenmp -fPIC -lm -Wall #-llikwid -DLIKWID_PERFMON

# LIBRARIES
LIBPARDISO=-lpardiso700-GNU840-X86-64-RINLA 	#expects the path in .bashrc (LD_LIBRARY_PATH, LIBRARY)
LIBMKL=-L${MKLROOT}/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core
#LIBRGF=-lrgf


#LIBCHOLMOD=-lcholmod
ARMADILLO=-larmadillo

LIBS=$(LIBPARDISO) $(LIBMKL) $(ARMADILLO) #$(LIBCHOLMOD)

# INCLUDES
INCEIGEN=-I/usr/include/eigen3
INCBFGS=-I/home/x_gaedkelb/applications/LBFGSpp/include
#INCCHOLMOD= -I/usr/include/suitesparse

INCLUDES=$(INCEIGEN) $(INCBFGS) #$(INCCHOLMOD)

RGF_INCLUDEDIR  = $(INCCUD) $(INCMAG)
RGF_LIBS	    = $(SCALAPACK) $(BLACS) $(LAPACK) $(BLAS) $(LINKS) $(OPENMP) $(CUDA) $(MAGMA) $(F90_LIBS)

RGF_CC_FILES   = RGF/Utilities.o #RGF/main.o
RGF_CU_FILES   = RGF/CWC_utility.o

all: call_INLA

#librgf.a: $(CC_FILES) $(CU_FILES)
#	ar rvs $@ $^

#librgf.so: $(CC_FILES) $(CU_FILES)
#	$(LOADER) --shared $(CC_FILES) $(CU_FILES) $(LOADFLAGS) $(LIBS) -lm -o $@ $(DEBUG) -fPIC

#RGF/main: $(RGF_CC_FILES) $(RGF_CU_FILES) 
#	$(RGF_LOADER) $^ $(RGF_LOADFLAGS) $(RGF_LIBS) $(OPENMP) -lm -o $@ $(DEBUG)

#RGF/main.o: RGF/main.C
#	$(CXX) -c $^ $(RGF_CXXFLAGS) $(DEBUG) $(RGF_FLAGS) $(RGF_INCLUDEDIR) -fPIC -o $@	

RGF/Utilities.o: RGF/Utilities.C
	$(RGF_LOADER) -c $^ $(RGF_CXXFLAGS) $(DEBUG) $(RGF_FLAGS) $(RGF_INCLUDEDIR) -fPIC -o $@

RGF/CWC_utility.o: RGF/CWC_utility.cu
	$(NVCC) -c $^ $(NVCCFLAGS) $(INCCUD) --compiler-options '-fPIC' -o $@

call_INLA: call_INLA.o PostTheta.o PardisoSolver.o RGFSolver.o RGF/Utilities.o RGF/CWC_utility.o #solver_cholmod.o 
	$(RGF_LOADER) $^ $(LOADFLAGS) $(RGF_LOADFLAGS) $(LIBS) $(RGF_LIBS) $(DEBUG) -lm -o $@  #$(LDFLAGS) $(LDLIBS) 

call_INLA.o: call_INLA.cpp
	$(CXX) -c $< $(CXXFLAGS) $(RGF_CXXFLAGS) $(RGF_FLAGS) $(INCLUDES) $(RGF_INCLUDEDIR) -o $@ 

%.o : %.cpp %.h
	$(CXX) -c $< $(CXXFLAGS) $(INCLUDES) -o $@ 

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



