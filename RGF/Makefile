# compile into shared library and then include this in main makefile
#
#####################
## PATHS TO BE ADAPTED

# usually home directory
BASE_PATH=/home/hpc/ihpc/ihpc060h



######################
# make sure MKLROOT is set
MKLHOME       = $(MKLROOT)
MKLHOME       = $(MKLROOT)/lib/intel64

LAPHOME       = $(MKLHOME)

include ./make.inc

## for nvtx ranges: lnvToolsExt
DEBUG       =  -lnvToolsExt #-O1 -g #-fsanitize=address,signed-integer-overflow #-Wall

# for mainEigen
#INCEIGEN=-I/usr/include/eigen3
INCEIGEN=-I$(BASE_PATH)/applications/eigen

ARMADILLO=-larmadillo

$(info ============== Library paths ==============)

$(info MKL    : $(MKLHOME))
$(info LAPACK : $(LAPACK))
$(info CUDA   : $(CUDA))
$(info MAGMA  : $(MAGMA))

INCLUDEDIR  = $(INCCUD) $(INCMAG) $(INCEIGEN)
LIBS	    = $(ARMADILLO) $(LAPACK) $(OPENMP) $(CUDA) $(MAGMA)  


$(info ============== Compiling ==============)

all: mainEigen #mainEigen

CC_FILES   = Utilities.o mainEigen.o #main.o
CU_FILES   = CWC_utility.o

mainEigen: $(CC_FILES) $(CU_FILES)
	$(RGF_MPICXX) $(RGF_CXXFLAGS) $(CC_FILES) $(CU_FILES) $(LIBS) $(DEBUG) -lm -o $@ # $(RGF_LOADFLAGS)

main: $(CC_FILES) $(CU_FILES)
	$(RGF_MPICXX) $(DEBUG) $(RGF_CXXFLAGS) $(CC_FILES) $(CU_FILES) $(OPENMP) $(LIBS) -lm -o $@ # $(RGF_LOADFLAGS)


.C.o:
	$(CXX) -c $< $(RGF_CXXFLAGS) $(RGF_FLAGS) $(INCLUDEDIR) $(DEBUG) $(OPENMP) -fPIC

CWC_utility.o: CWC_utility.cu
	$(NVCC) -c CWC_utility.cu $(NVCCFLAGS) $(INCCUD) --compiler-options '-fPIC' 

clean:	
	rm -f *.o RGFSolver
	rm -f mainEigen
	rm -f main

