
MKLROOT?=$(shell echo $$MKLROOT)

DEBUG= -g

BASE_PATH=/home/hpc/ihpc/ihpc060h

MKLHOME       = $(MKLROOT)/lib/intel64

LIBMKL=-L${MKLROOT}/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core
ARMADILLO=-larmadillo

#INCEIGEN=-I${BASE_PATH}/applications/eigen-3.4.0
INCEIGEN=-I${BASE_PATH}/applications/eigen


INCMAG            = -I$(BASE_PATH)/applications/magma-2.7.2/include
MAGMA             = -L$(BASE_PATH)/applications/magma-2.7.2/lib -lmagma

INCCUD	          = -I$(CUDAHOME)/include
CUDA              = -L/usr/lib64/ -L$(CUDAHOME)/lib64 -lcudart -lcublas -lcufft -lcusparse -lcusolver -Wl,-rpath,$(CUDAHOME)/lib64


INCLUDEDIR  = $(INCCUD) $(INCMAG) $(INCEIGEN)
LIBS	    = $(LIBMKL) $(ARMADILLO) $(CUDA) $(MAGMA)  



main: main_MPOTRF.C   # main_MPOTRF.C 
	g++ $< -o $@ $(LIBS) $(INCLUDEDIR) $(DEBUG) -fopenmp


clean:
	rm -rf main
