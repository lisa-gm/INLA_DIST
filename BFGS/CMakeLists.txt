
cmake_minimum_required(VERSION 3.5)

project(b_INLA
	DESCRIPTION "simplified core algorithm of INLA"
	VERSION 0.5
	LANGUAGES CXX C Fortran)

# Fortran\CPP
include(FortranCInterface)
FortranCInterface_VERIFY(CXX)
unset(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES)


set(MKL_DIR $ENV{MKLROOT})

set(PARDISO_LIB "/home/x_gaedkelb/applications/pardiso/libpardiso700-GNU840-X86-64-RINLA.so")

set(CMAKE_CXX_FLAGS "-lm -fopenmp -lpthread -lm -lgomp -lgfortran -fopenmp -fPIC -lm")

# compiler flags
#set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS}-w -O3 -fPIC -D_LONG_INTEGER_ -std=c++11")
# g++ $< -o $@ -I/usr/include/eigen3 -lm -I/home/x_gaedkelb/applications/LBFGSpp/include -I/usr/include/suitesparse -lcholmod -lm \
#	-fopenmp \
#	/home/x_gaedkelb/applications/pardiso/libpardiso700-GNU840-X86-64-RINLA.so -L${MKLROOT}/lib/intel64 -lmkl_gf_lp64 
# -lmkl_gnu_thread -lmkl_core -lpthread -lm -lgomp -lgfortran -fPIC  -fopenmp -fPIC -lm \
	
# EIGEN

set(EIGEN_DIR usr/include/eigen3)
#include_directories(include 
#					${EIGEN_DIR})

# BFGS
set(BFGS_DIR /home/x_gaedkelb/applications/LBFGSpp)
include_directories(include
					${BFGS_DIR}/include)

# CHOLMOD 

# SUITESPARSE for CHOLMOD
set(CHOLMOD_DIR "/home/x_gaedkelb/applications/SuiteSparse")
#set(CHOLDMOD_DIR "/usr/")

include_directories(include
					${CHOLMOD_DIR}/include)

find_library(cholmod ${CHOLMOD_DIR}/lib/libcholmod.a)


# -------------- EXECUTABLE ------------- #

add_executable(call_INLA call_INLA.cpp theta_function_test.cpp PardisoSolver.cpp ) # solver_cholmod.cpp

# CHOLMOD
target_link_libraries(call_INLA cholmod)

# MKL
find_library(libmkl_gf_lp64 
	NAMES libmkl_gf_lp64.a
	PATHS ${MKL_DIR} ${MKL_DIR}/lib ${MKL_DIR}/mkl/lib ${MKL_DIR}/lib/intel64 
	NO_DEFAULT_PATH
	) 
find_library(libmkl_gnu_thread
	NAMES libmkl_gnu_thread.a
	PATHS ${MKL_DIR} ${MKL_DIR}/lib ${MKL_DIR}/mkl/lib ${MKL_DIR}/lib/intel64 
	NO_DEFAULT_PATH
	)

find_library(libmkl_core
	NAMES libmkl_core.a 
	PATHS ${MKL_DIR} ${MKL_DIR}/lib ${MKL_DIR}/mkl/lib ${MKL_DIR}/lib/intel64 
	NO_DEFAULT_PATH
	) 

#target_link_libraries(call_INLA ${EIGEN_DIR})
target_include_directories(call_INLA BEFORE PRIVATE "$<1:/usr/include/eigen3>")


target_link_libraries(call_INLA -Wl,--start-group ${libmkl_gf_lp64} ${libmkl_gnu_thread} ${libmkl_core} -Wl,--end-group)

# PARDISO
target_link_libraries(call_INLA ${PARDISO_LIB})
