
MKLROOT?=$(shell echo $$MKLROOT)

############# RGF ################

MKLHOME       = $(MKLROOT)/lib/intel64
BLAHOME       = $(MKLHOME)
LAPHOME       = $(MKLHOME)
SCAHOME       = $(MKLHOME)

RGF_PATH      = ../RGF
include $(RGF_PATH)/make.inc
#############################################################################################################

CXX=mpic++
CXXFLAGS=-lpthread -lm -lgomp -lgfortran -fPIC -fopenmp -fPIC -lm -Wall #-llikwid -DLIKWID_PERFMON

# LIBRARIES
# IBEX : -lpardiso720-GNU840-X86-64-RINLA, HAAVARD'S SERVER : -lpardiso700-GNU840-X86-64-RINLA
LIBPARDISO=-lpardiso700-GNU840-X86-64-RINLA 	#expects the path in .bashrc (LD_LIBRARY_PATH, LIBRARY)
LIBMKL=-L${MKLROOT}/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core

#LIBCHOLMOD=-lcholmod

LIBS=$(LIBPARDISO) $(LIBMKL) #$(LIBCHOLMOD)

# INCLUDES
INCEIGEN=-I/usr/include/eigen3
INCBFGS=-I/home/x_gaedkelb/applications/LBFGSpp/include
#INCCHOLMOD= -I/usr/include/suitesparse

INCLUDES=$(INCEIGEN) $(INCBFGS) #$(INCCHOLMOD)


RGF_INCLUDEDIR  = $(INCCUD) $(INCMAG)
RGF_LIBS	    = $(LAPACK) $(RGF_LINKS) $(CUDA) $(MAGMA) 

RGF_CC_FILES   = $(RGF_PATH)/Utilities.o #RGF/main.o
RGF_CU_FILES   = $(RGF_PATH)/CWC_utility.o

all: call_INLA

#librgf.a: $(CC_FILES) $(CU_FILES)
#	ar rvs $@ $^

#librgf.so: $(CC_FILES) $(CU_FILES)
#	$(LOADER) --shared $(CC_FILES) $(CU_FILES) $(LOADFLAGS) $(LIBS) -lm -o $@ $(DEBUG) -fPIC

#RGF/main: $(RGF_CC_FILES) $(RGF_CU_FILES) 
#	$(RGF_LOADER) $^ $(RGF_LOADFLAGS) $(RGF_LIBS) -lm -o $@ $(DEBUG)

#RGF/main.o: RGF/main.C
#	$(CXX) -c $^ $(RGF_CXXFLAGS) $(DEBUG) $(RGF_FLAGS) $(RGF_INCLUDEDIR) -fPIC -o $@	

$(RGF_PATH)/Utilities.o: $(RGF_PATH)/Utilities.C
	$(CXX) -c $^ $(RGF_CXXFLAGS) $(DEBUG) $(RGF_FLAGS) $(RGF_INCLUDEDIR) -fPIC -o $@

$(RGF_PATH)/CWC_utility.o: $(RGF_PATH)/CWC_utility.cu
	$(NVCC) -c $^ $(NVCCFLAGS) $(INCCUD) --compiler-options '-fPIC' -o $@


call_INLA: call_INLA.cpp PostTheta.o Model.o PardisoSolver.o RGFSolver.o $(RGF_CC_FILES) $(RGF_CU_FILES) #solver_cholmod.o 
	$(CXX) $^ -o $@  $(INCLUDES) $(LIBS) $(CXXFLAGS) $(LOADFLAGS) $(RGF_FLAGS) $(RGF_LIBS) -lm #$(LDFLAGS) $(LDLIBS) 

%.o : %.cpp %.h
	$(CXX) $(LIBS) -c $< -o $@ $(INCLUDES) $(CXXFLAGS) 

#RGF/librgf.a:
#	$(MAKE) -C RGF

#RGF/librgf.so:
#	$(MAKE) -C RGF


#clean:
#	rm -f call_INLA
#	rm -f *.o

clean:	
	rm -f *.o call_INLA
	rm -f RGF/*.o
	#rm -f librgf.so
	#rm -f librgf.a



