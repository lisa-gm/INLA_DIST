# compile into shared library and then include this in main makefile

#include ./make.inc_kaust
#         $(CXX) -c $< $(CXXFLAGS) $(DEBUG) $(OPENMP) $(FLAGS) $(INCLUDEDIR)
include ./make.inc

MKLHOME       = $(MKLROOT)/lib/intel64
BLAHOME       = $(MKLHOME)
LAPHOME       = $(MKLHOME)
SCAHOME       = $(MKLHOME)

DEBUG       =   #-Wall # -g -fsanitize=address,signed-integer-overflow

ARMADILLO   = -larmadillo

INCLUDEDIR  = $(INCCUD) $(INCMAG)
LIBS	    = $(ARMADILLO) $(SCALAPACK) $(BLACS) $(LAPACK) $(BLAS) $(LINKS) $(OPENMP) $(CUDA) $(MAGMA) $(F90_LIBS)

CC_FILES   = Utilities.o main.o #intermed_fct_RGF.o
CU_FILES   = CWC_utility.o

#librgf.a: $(CC_FILES) $(CU_FILES)
#	ar rvs $@ $^

#librgf.so: $(CC_FILES) $(CU_FILES)
#	$(LOADER) --shared $(CC_FILES) $(CU_FILES) $(LOADFLAGS) $(LIBS) -lm -o $@ $(DEBUG) -fPIC

main: $(CC_FILES) $(CU_FILES)
	$(RGF_LOADER) $(CC_FILES) $(CU_FILES) $(RGF_LOADFLAGS) $(OPENMP) $(LIBS) -lm -o $@ $(DEBUG)

.C.o:
	$(CXX) -c $< $(RGF_CXXFLAGS) $(DEBUG) $(RGF_FLAGS) $(INCLUDEDIR) $(OPENMP) #-fPIC

CWC_utility.o: CWC_utility.cu
	$(NVCC) -c CWC_utility.cu $(NVCCFLAGS) $(INCCUD) #--compiler-options '-fPIC' 

clean:	
	rm -f *.o RGFSolver
	rm -f main
	#rm -f librgf.so
	#rm -f librgf.a
