# b_INLA

## Overview of the different directories

### 1.) BFGS

Nested OpenMP implementation. Using OpenMP for the parallelisation of the gradient (ideally using 9 threads for dim(theta)=4) and using multiple threads in PARDISO (ideally 16 or 32 threads). The Hessian computation is also using OpenMP tasks, here we ideally have 33 threads, as there are 33 function evaluations to perform for dim(theta) = 4.  

#### 2.) MPI_BFGS

Uses nested OpenMP parallelism and MPI. Code structured differently. Master - worker setup. Master runs main part of the code. The workers set up a theta function evaluation (its all in Model.cpp) and wait for master to send them a theta value to compute the corresponding f(theta) and return that value. Drawback, we need one more node than there are "actual" jobs i.e. function evaluations to do. Nested OpenMP parallelism is now inside the Model class. We split the factorisation of the prior precision matrix of the random effects $Q_{st}$, which is a large tridiagonal block matrix (just spatial-temporal effects) whose log determinant we require and the factorisation (and solve) of the conditional precision matrix $Q_{x|y}$ where $x$ are the latent parameters and $y$ the observations. Then for each of these we can again run PARDISO with e.g. 16 threads. 

#### 3.) MPI_all_BFGS

Similar to MPI_BFGS, however, here each rank performs all the "outer" computations, the function evaluations are assigned to individual ranks. This makes it a more flexible framework in terms of number of ranks in (2) we needed exactly the right amount of ranks as there were function evaluations. Now any number of ranks is possible. At the end of the function evaluations, all information is shared using MPI_Allreduce. Hence, all ranks go through the BFGS search, but also all ranks perform function evaluations. 

#### 4.) RGF

RGF folder that contains all the code required for the GPU Solver. (1)-(3) all access the same RGF code. Currently compiled as object files with headers. Maybe later it can be compiled into a shared library to make it more similar to PARDISO.

#### 5.) Contains a whole bunch of example scripts to test individual compenents, develop the MPI implementation, etc.

