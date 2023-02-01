# Block Tridiagonal Arrowhead Solver

subdirectory that contains everything related to BTA solver which has a standalone callable main. 
Main operations that can be performed:

- Cholesky decomposition (GPU)
- Forward-backward solve using Cholesky factor (CPU)
- selected matrix inversion (GPU, exports diagonal of the inverse to CPU

Assumes matrix to be in right format, $n_t$ defines number of large diagonal blocks of size $n_s \times n_s$, $n_b$ number of rows/columns in arrowhead structure. Overall matrix size is thus $n = n_s \cdot n_t + n_b$. 

## Installation

requires CUDA, MAGMA, Eigen, Armadillo (just for 1 read file operation, maybe this can be replaced). All paths related to GPU code are in `make.inc` (base_path of the user has to be set at the top of the file), rest in `Makefile`. Here MKLROOT & LAPACK path need to be set/known. 


## Running mainEigen.C

contains if-statement section that just generates dummy matrix of small size (some parameters can be changed, not all) and can be run as 

`./mainEigen `

without any additional inputs. When the else-section is active requires external .dat files to read in that have to follow specific naming policy. Calls look something like this

`srun ./main ${folder_path} ${ns} ${nt} ${nb} ${no} `

where folder_path describes the path to the data files, ns, nb and nt as above and no describes the number of observations. There are numerous run scripts e.g. in `run_script.sh` that can be adapted to automate this process. 


