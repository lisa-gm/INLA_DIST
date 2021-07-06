
cmake_minimum_required(VERSION 3.5)

project(b_INLA)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
 

# -------------- CHOLMOD ------------- #

# SUITESPARSE for CHOLMOD
#set (CHOLMOD_DIR "/home/x_gaedkelb/applications/SuiteSparse")
set(CHOLDMOD_DIR "/usr/")

include_directories(include
					${CHOLMOD_DIR}/include)

find_library(cholmod ${CHOLMOD_DIR}/lib/libcholmod.a)



# -------------- BFGS ------------- #
set(BFGS_DIR /home/x_gaedkelb/applications/LBFGSpp)
include_directories(include
					 ${BFGS_DIR}/include)

# -------------- EXECUTABLE ------------- #

add_executable(example example.cpp theta_function.cpp solver.cpp)

# -------------- LINKING ------------- #

target_link_libraries (example Eigen3::Eigen)

target_link_libraries (example cholmod)