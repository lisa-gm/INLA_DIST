#  INLA<sub>DIST</sub>

Repository to host INLA<sub>DIST</sub>. This code base provides a highly scalable approach to Bayesian spatial-temporal modeling. It relies on the methodology of integrated nested Laplace approximations, in combination with the stochastic partial differential equations approach to provide an efficient framework for performing inference, leveraging sparse representations of the underlying processes. Building upon the INLA-SPDE approach, the main focus of the current implementation is on non-separable spatial-temporal model, derived from a partial differential equation equation. This family of physically inspired models captures the spatial-temporal behavior of the underlying physical process. We tackle the challenge of higher model complexity by developing a solution method which exploits the underlying model sparsity, utilizes parallelism and leverages the strengths of modern compute architectures. Details can be found in [Integrated Nested Laplace Approximations for Large-Scale Spatial-Temporal Bayesian Modeling](https://arxiv.org/abs/2303.15254).

## Overview of the different directories

main directories which contains the majority of the source code of our method and in particular everything related to the INLA methodology:
  - INLA<sub>main</sub>    : most general version, linear solver can be chosen at runtime.
  - INLA<sub>CPUonly</sub>  : CPU only version using PARDISO solver.
  - INLA<sub>predict</sub> : new version under development with additional prediction features and accuracy measures.

other directories:
  - BTA   : contains all code related to the block tridiagonal arrowhead solver.
  - Rscripts     : scripts for data preprocessing & generation of synthetic datasets. contains code to export to c-readable files.
  - Test Scripts : small test scripts to develop and test features in the code. for development purposes.

Detailed documentation of the code can be found [here](https://lisa-gm.github.io/INLA_DIST/documentation/html/index.html).

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

################################################################################################

### Running 

Please check the respective subfolders for detailed instructions depending on what you would like to do.


