# make.inc for BTA solver

## check if any of the following paths need to be changed!!
## set path for CUDA, MAGMA, ...


###### CPU ######
BTA_MPICXX        = g++ #mpicxx #-Wl,-rpath -Wl,$(INTEL64)/compiler/lib/intel64

BTA_FLAGS         = -DAdd_ -DSOLVE_CRAY -D__MINMAX -Wno-write-strings
BTA_CXXFLAGS      = -O3 -DMPICH_IGNORE_CXX_SEEK #-O3 -ffast-math -funroll-loops -std=c++14
#BTA_CXXFLAGS      = -g -DMPICH_IGNORE_CXX_SEEK 


#LAPACK            = -L$(LAPHOME) -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -Wl,-rpath -Wl,$(MKLHOME) 
OPENBLAS           = -L$(OPENBLASDIR)/lib -lopenblas -Wl,-rpath,$(OPENBLASDIR)/lib
OPENMP             = -fopenmp

###### GPU ######
NVCC              = nvcc
BTA_NVCCFLAGS     = -arch=compute_90 -code=sm_90

BTA_CUDA_SDK      = 3
BTA_COMMONFLAGS   = -DCUDA_SDK=$(CUDA_SDK)

INCMAG            = -I$(BASE_PATH)/applications/magma-2.9.0/include
MAGMA             = -L$(BASE_PATH)/applications/magma-2.9.0/magma/lib -lmagma

CUDAHOME          = $(CUDA_HOME)
INCCUD	          = -I$(CUDAHOME)/include
CUDA              = -L/usr/lib64/ -L$(CUDAHOME)/lib64 -lcudart -lcublas -lcufft -lcusparse -lcusolver -Wl,-rpath,$(CUDAHOME)/lib64



