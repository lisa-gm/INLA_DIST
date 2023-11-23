#  $\text{INLA}_{\text{DIST}}$

Repository to host $\text{INLA}_{\text{DIST}}$. This code base provides a highly scalable approach to Bayesian spatial-temporal modeling. It relies on the methodology of integrated nested Laplace approximations, in combination with the stochastic partial differential equations approach to provide an efficient framework for performing inference, leveraging sparse representations of the underlying processes. Building upon the INLA-SPDE approach, the main focus of the current implementation is on non-separable spatial-temporal model, derived from a partial differential equation equation. This family of physically inspired models captures the spatial-temporal behavior of the underlying physical process. We tackle the challenge of higher model complexity by developing a solution method which exploits the underlying model sparsity, utilizes parallelism and leverages the strengths of modern compute architectures. Details can be found in [Integrated Nested Laplace Approximations for Large-Scale Spatial-Temporal Bayesian Modeling](https://arxiv.org/abs/2303.15254).

## Overview of the different directories

main directories which contains the majority of the source code of our method and in particular everything related to the INLA methodology:
  - $\text{INLA}_{\text{MAIN}}$    : most general version, linear solver can be chosen at runtime.
  - $\text{INLA}_{\text{CPUonly}}$ : CPU only version using PARDISO solver.
  - $\text{INLA}_{\text{predict}}$ : new version under development with additional prediction features and accuracy measures.

other directories:
  - $\text{BTA}$ : contains all code related to the block tridiagonal arrowhead solver.
  - Rscripts     : scripts for data preprocessing & generation of synthetic datasets. contains code to export to c-readable files.
  - Test Scripts : small test scripts to develop and test features in the code. for development purposes. 

#### 4.) $\text{INLA}_{\text{MAIN}}$

```main file: call_INLA.cpp```
"default" GPU version. 1 MPI rank for each function evaluation (like MPI_all_BFGS) and then OpenMP parallelism for nominator/denominator.



#### 3.) MPI_all_BFGS

Similar to MPI_BFGS, however, here each rank performs all the "outer" computations, the function evaluations are assigned to individual ranks. This makes it a more flexible framework in terms of number of ranks in (2) we needed exactly the right amount of ranks as there were function evaluations. Now any number of ranks is possible. At the end of the function evaluations, all information is shared using MPI_Allreduce. Hence, all ranks go through the BFGS search, but also all ranks perform function evaluations. 

When sufficiently many ranks are available, each function evaluation is assigned two processes, 1 for nominator, 1 for denominator, this way each PARDISO instance can have their own node. 



#### 5.) $\text{BTA}$

RGF folder that contains all the code required for the **B**lock**T**ridiagonal **A**rrowhead (GPU-based) Solver. (1)-(3) all access the same RGF code. Currently compiled as object files with headers. Alternatively [PARDISO](https://panua.ch/pardiso/) is used as a CPU-based sparse linear solver.

#### 5.) RScripts



#### 5.) Test Scripts

Contains a whole bunch of example scripts to test individual compenents, develop the MPI implementation, etc.

################################################################################################

### Installation

The implementation targets linux-based systems. Compile from the respective subfolder you are interested in using the provided ``makefile``, potentially adapting some of the provided paths. The following additional software libraries or packages are needed:

- [Eigen](https://eigen.tuxfamily.org)
- [Armadillo](https://arma.sourceforge.net/) (only for reading in matrices in CSR format)

#### INLA-SPDE optimization routine
- [adapted LBFGSpp](https://github.com/lisa-gm/adapted_LBFGSpp)

Calls either the BTA solver or PARDISO, depending on the choice of linear solver.

#### BTA Solver

- [MAGMA](https://icl.utk.edu/magma/)
- [CUDA](https://developer.nvidia.com/cuda-toolkit)

The BTA solver can be compiled independently from the RGF subdirectory. It has its own ``makefile`` and a ``mainEigen.C`` which can be executed using one the provided run scripts.
