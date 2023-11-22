# Block Tridiagonal Arrowhead Solver

subdirectory that contains everything related to BTA solver which has a standalone callable main otherwise it gets called from call_INLA.cpp from the develop subfolder, which is one of the main executables of the repository. 
Main operations that can be performed:

- Cholesky decomposition (GPU)
- Forward-backward solve using Cholesky factor (CPU)
- selected matrix inversion (GPU)
  
Assumes matrix to be in right format, $n_t$ defines number of large diagonal blocks of size $n_s \times n_s$, $n_b$ number of rows/columns in arrowhead structure. Overall matrix size is thus $n = n_s \cdot n_t + n_b$. 

## Installation

requires CUDA, MAGMA, Eigen, Armadillo (just for 1 read file operation). All paths related to GPU code are in `make.inc` (base_path of the user has to be set at the top of the file), rest in `Makefile`. Here MKLROOT & LAPACK path need to be set or known. The same holds for CUDA (set ```CUDAHOME```), MAGMA (set ```MAGMA```), Eigen (set ```INCEIGEN``` (header only) and Armadillo.  


## Running mainEigen.C

The executable requires information of where to find the data, including the FEM matrices of the spatial-temporal mesh. Examples of how this needs to be provided can be found in one of the different run scripts, for instance run_script.sh. It looks something like 

```
./mainEigen ${ns} ${nt} ${nb} ${no} ${folder_path} ${solver_type}
```

Otherwise there is also if-statement section that just generates dummy matrix of small size (some parameters can be changed, not all).  

## Additional Information

We would like to express our gratitude for the initial software support on the selected block inversion from Prof. Mathieu Luisier. The [OMEN[^1]](https://doi.org/10.1109/NANO.2008.110) software infrastructure was used as a starting point to derive and implement the BTA solver.

[^1]: M. Luisier and G. Klimeck, Omen an atomistic and full-band quantum transport simulator for post-CMOS nanodevices, in 2008 8th IEEE Conference on Nanotechnology, IEEE, 2008, pp. 354–357. 10.1109/NANO.2008.110.



