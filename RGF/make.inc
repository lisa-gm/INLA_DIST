# make.inc for RGF solver

BASE_PATH=/home/x_gaedkelb
#BASE_PATH=/home/hpc/ihpc/ihpc060h

###### CPU ######
RGF_MPICXX        = mpicxx #-Wl,-rpath -Wl,$(INTEL64)/compiler/lib/intel64

RGF_FLAGS         = -DAdd_ -DSOLVE_CRAY -D__MINMAX -Wno-write-strings
#RGF_CXXFLAGS      = -O3 -ffast-math -funroll-loops -DMPICH_IGNORE_CXX_SEEK #-O3
RGF_CXXFLAGS      = -g -DMPICH_IGNORE_CXX_SEEK 


LAPACK            = -L$(LAPHOME) -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -Wl,-rpath -Wl,$(MKLHOME) 
OPENMP            = -fopenmp

###### GPU ######
NVCC              = nvcc
#RGF_NVCCFLAGS     = -arch=compute_70 -code=sm_70
RGF_NVCCFLAGS     = -arch=compute_80 -code=sm_80

RGF_CUDA_SDK      = 3
RGF_COMMONFLAGS   = -DCUDA_SDK=$(CUDA_SDK)

#INCMAG            = -I/home/x_gaedkelb/applications/magma-2.5.4/include
#MAGMA             = -L/home/x_gaedkelb/applications/magma-2.5.4/lib -lmagma
INCMAG            = -I$(BASE_PATH)/applications/custom_magma/magma/include
MAGMA             = -L$(BASE_PATH)/applications/custom_magma/magma/lib -lmagma


CUDAHOME          = /usr/local/cuda
#CUDAHOME          = $(CUDA_HOME)
INCCUD	          = -I$(CUDAHOME)/include
CUDA              = -L/usr/lib64/ -L$(CUDAHOME)/lib64 -lcudart -lcublas -lcufft -lcusparse -lcusolver -Wl,-rpath,$(CUDAHOME)/lib64
# CUDA              = -L/usr/lib64/ -L$(CUDAHOME)/lib64 -lcudart -lcublas -lcufft -lcusparse -Wl,-rpath,$(CUDAHOME)/lib64


# =====================================================

#RGF_MPICXX        = mpic++ #-Wl,-rpath -Wl,$(INTEL64)/compiler/lib/intel64

#RGF_LOADER        = mpic++ -Wl,-rpath -Wl,$(INTEL64)/compiler/lib/intel64
#F90_LIBS          = -L$(GNU64)/lib64 -lgfortran -Wl,-rpath,L$(GNU64)/lib64
#LINKS             = -Wl,-rpath -Wl,$(GNU64)/lib64
#RGF_LOADFLAGS     = -ffast-math -funroll-loops -L$(GNU64)/lib64 -lgfortran -L$(INTEL64)/compiler/lib/intel64 #-O3 -L$(MPIHOME)/lib/ 

#export LD_LIBRARY_PATH=$(GNU64)/lib64:$(INTEL64)/lib

