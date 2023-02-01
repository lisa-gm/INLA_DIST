# Block Tridiagonal Arrowhead Solver

subdirectory that contains everything related to BTA solver which has a standalone callable main. 
Main operations that can be performed:

- Cholesky decomposition (GPU)
- Forward-backward solve using Cholesky factor (CPU)
- selected matrix inversion (GPU, exports diagonal of the inverse to CPU

Assumes matrix to be in right format, $n_t$ defines number of large diagonal blocks of size $n_s \times n_s$, $n_b$ number of rows/columns in arrowhead structure. Overall matrix size is thus $n = n_s \cdot n_t + n_b$. 
