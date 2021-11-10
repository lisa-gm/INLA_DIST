# make.inc for RGF solver

RGF_FLAGS         = -DAdd_ -DSOLVE_CRAY -D__MINMAX -Wno-write-strings

RGF_LOADER        = mpic++ -Wl,-rpath -Wl,$(INTEL64)/compiler/lib/intel64
RGF_MPICXX        = mpic++ -Wl,-rpath -Wl,$(INTEL64)/compiler/lib/intel64

RGF_CXXFLAGS      = -O3 -ffast-math -funroll-loops -DMPICH_IGNORE_CXX_SEEK
RGF_LOADFLAGS     = -O3 -ffast-math -funroll-loops -L$(GNU64)/lib64 -lgfortran -L$(INTEL64)/compiler/lib/intel64 -L$(MPIHOME)/lib/ 

NVCC              = nvcc
RGF_NVCCFLAGS     = -arch=compute_70 -code=sm_70

RGF_CUDA_SDK      = 3
RGF_COMMONFLAGS   = -DCUDA_SDK=$(CUDA_SDK)

F90_LIBS          = -L$(GNU64)/lib64 -lgfortran -Wl,-rpath,L$(GNU64)/lib64
LAPACK            = -L$(LAPHOME) -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -Wl,-rpath -Wl,$(MKLHOME)

LINKS             = -Wl,-rpath -Wl,$(GNU64)/lib64
OPENMP            = -fopenmp
export LD_LIBRARY_PATH=$(GNU64)/lib64:$(INTEL64)/lib

INCMAG            = -I/home/x_gaedkelb/applications/magma-2.5.4/include
MAGMA             = -L/home/x_gaedkelb/applications/magma-2.5.4/lib -lmagma

CUDAHOME          = /usr/local/cuda
INCCUD	          = -I$(CUDAHOME)/include
CUDA              = -L/usr/lib64/ -lcuda -L$(CUDAHOME)/lib64 -lcudart -lcublas -lcufft -lcusparse -Wl,-rpath,$(CUDAHOME)/lib64



