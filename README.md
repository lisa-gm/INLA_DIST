# b_INLA

Repository to host $\text{INLA}_{\text{DIST}}$. This code base provides a highly scalable approach to Bayesian spatial-temporal modeling. It relies on the methodology of integrated nested Laplace approximations, in combination with the stochastic partial differential equations approach to provide an efficient framework for performing inference, leveraging sparse representations of the underlying processes. Building upon the INLA-SPDE approach, the main focus of the current implementation is on non-separable spatial-temporal model, derived from a partial differential equation equation. This family of physically inspired models captures the spatial-temporal behavior of the underlying physical process. We tackle the challenge of higher model complexity by developing a solution method which exploits the underlying model sparsity, utilizes parallelism and leverages the strengths of modern compute architectures. Details can be found in [Integrated Nested Laplace Approximations for Large-Scale Spatial-Temporal Bayesian Modeling](https://arxiv.org/abs/2303.15254).

## Overview of the different directories

The subdirectories MPI_all_BFGS, develop and predict are 

### 1.) BFGS

Nested OpenMP implementation. Using OpenMP for the parallelisation of the gradient (ideally using 9 threads for dim(theta)=4) and using multiple threads in PARDISO (ideally 16 or 32 threads). The Hessian computation is also using OpenMP tasks, here we ideally have 33 threads, as there are 33 function evaluations to perform for dim(theta) = 4.  

#### 2.) MPI_BFGS

Uses nested OpenMP parallelism and MPI. Code structured differently. Master - worker setup. Master runs main part of the code. The workers set up a theta function evaluation (its all in Model.cpp) and wait for master to send them a theta value to compute the corresponding f(theta) and return that value. Drawback, we need one more node than there are "actual" jobs i.e. function evaluations to do. Nested OpenMP parallelism is now inside the Model class. We split the factorisation of the prior precision matrix of the random effects $Q_{st}$, which is a large tridiagonal block matrix (just spatial-temporal effects) whose log determinant we require and the factorisation (and solve) of the conditional precision matrix $Q_{x|y}$ where $x$ are the latent parameters and $y$ the observations. Then for each of these we can again run PARDISO with e.g. 16 threads. 

#### 3.) MPI_all_BFGS

Similar to MPI_BFGS, however, here each rank performs all the "outer" computations, the function evaluations are assigned to individual ranks. This makes it a more flexible framework in terms of number of ranks in (2) we needed exactly the right amount of ranks as there were function evaluations. Now any number of ranks is possible. At the end of the function evaluations, all information is shared using MPI_Allreduce. Hence, all ranks go through the BFGS search, but also all ranks perform function evaluations. 

When sufficiently many ranks are available, each function evaluation is assigned two processes, 1 for nominator, 1 for denominator, this way each PARDISO instance can have their own node. 


#### 4.) develop 

"default" GPU version. 1 MPI rank for each function evaluation (like MPI_all_BFGS) and then OpenMP parallelism for nominator/denominator.


#### 5.) RGF

RGF folder that contains all the code required for the **B**lock**T**ridiagonal **A**rrowhead (GPU-based) Solver. (1)-(3) all access the same RGF code. Currently compiled as object files with headers. Alternatively [PARDISO](https://panua.ch/pardiso/) is used as a CPU-based sparse linear solver.

#### 5.) Sample Scripts

Contains a whole bunch of example scripts to test individual compenents, develop the MPI implementation, etc.

################################################################################################

### Installation

The implementation targets linux-based systems. Compile from the respective subfolder you are interested in using the provided ``makefile``, potentially adapting some of the provided paths. The following additional software libraries or packages are needed:

- [Eigen](https://eigen.tuxfamily.org)
- [Armadillo](https://arma.sourceforge.net/) (only for reading in matrices in CSR format)

#### INLA-SPDE optimization routine
- [LBFGSpp](https://github.com/yixuan/LBFGSpp)

Will build and then call either the BTA solver or PARDISO, depending on the choice of linear solver.

#### BTA Solver

- [MAGMA](https://icl.utk.edu/magma/)
- [CUDA](https://developer.nvidia.com/cuda-toolkit)

The BTA solver can be compiled independently from the RGF subdirectory. It has its own ``makefile`` and a ``mainEigen.C`` which can be executed using one the provided run scripts.
