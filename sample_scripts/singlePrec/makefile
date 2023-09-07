
MKLROOT?=$(shell echo $$MKLROOT)

BASE_PATH=/home/x_gaedkelb
#BASE_PATH=/home/hpc/ihpc/ihpc060h

MKLHOME       = $(MKLROOT)/lib/intel64

LIBMKL=-L${MKLROOT}/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core
ARMADILLO=-larmadillo

INCEIGEN=-I${BASE_PATH}/applications/eigen-3.4.0

INCMAG            = -I$(BASE_PATH)/applications/magma-2.7.2/include
MAGMA             = -L$(BASE_PATH)/applications/magma-2.7.2/lib -lmagma

#CUDAHOME          = /usr/local/cuda
CUDAHOME	 	  = /usr/lib/cuda
#CUDAHOME          = $(CUDA_HOME)
INCCUD	          = -I$(CUDAHOME)/include
CUDA              = -L/usr/lib64/ -L$(CUDAHOME)/lib64 -lcudart -lcublas -lcufft -lcusparse -lcusolver -Wl,-rpath,$(CUDAHOME)/lib64


INCLUDEDIR  = $(INCCUD) $(INCMAG) $(INCEIGEN)
LIBS	    = $(LIBMKL) $(ARMADILLO) $(CUDA) $(MAGMA)  



main: main.C
	g++ $< -o $@ $(LIBS) $(INCLUDEDIR) $(DEBUG) -fopenmp


clean:
	rm -rf main
