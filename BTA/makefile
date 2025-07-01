# compile into shared library and then include this in main makefile
#
#####################
## PATHS TO BE ADAPTED

# usually home directory
BASE_PATH=$(HOME)

#####################

include ./make.inc

CXXFLAGS      = -O3 -march=native -ffast-math -funroll-loops -DMPICH_IGNORE_CXX_SEEK #-O3 -std=c++14 

## for nvtx ranges: lnvToolsExt
DEBUG       =  -lnvToolsExt #-O1 -g #-fsanitize=address,signed-integer-overflow #-Wall

# for mainEigen
INCEIGEN=-I$(BASE_PATH)/applications/eigen


$(info ============== Library paths ==============)

$(info OPENBLAS : $(OPENBLAS))
$(info CUDA   : $(CUDA))
$(info MAGMA  : $(MAGMA))

INCLUDEDIR  = $(INCCUD) $(INCMAG) $(INCEIGEN)
LIBS	    = $(OPENBLAS) $(OPENMP) $(CUDA) $(MAGMA)  


$(info ============== Compiling ==============)

all: mainEigen #mainEigen

CC_FILES   = Utilities.o mainEigen.o #main.o
CU_FILES   = CWC_utility.o

mainEigen: $(CC_FILES) $(CU_FILES)
	$(BTA_MPICXX) $(CXXFLAGS) $(CC_FILES) $(CU_FILES) $(LIBS) $(DEBUG) -lm -o $@ # $(BTA_LOADFLAGS)

# main: $(CC_FILES) $(CU_FILES)
# 	$(BTA_MPICXX) $(DEBUG) $(CXXFLAGS) $(CC_FILES) $(CU_FILES) $(OPENMP) $(LIBS) -lm -o $@ # $(BTA_LOADFLAGS)


.C.o:
	$(CXX) -c $< $(CXXFLAGS) $(BTA_FLAGS) $(INCLUDEDIR) $(DEBUG) $(OPENMP) -fPIC

CWC_utility.o: CWC_utility.cu
	$(NVCC) -c CWC_utility.cu $(BTA_NVCCFLAGS) $(BTA_CXXFLAGS) $(INCCUD) --compiler-options '-fPIC' 

clean:	
	rm -f *.o BTASolver
	rm -f mainEigen
	rm -f main

