#ifndef RGFSOLVER_H
#define RGFSOLVER_H

#include "mpi.h"

#include "Solver.h"

#if 0
typedef CPX T;
#define assign_T(val) CPX(val, 0.0)
#else
typedef double T;
#define assign_T(val) val
#endif

//extern "C" double 

 /**
 * @brief creates solver class using RGF-GPU for factorising, solving and selectively inverting linear system.
 * @details divided into set up, symbolic factorisation, numerical factorisation, numerical factorisation & solve 
 * and selected inversion (of the diagonal elements)
 * @note in each RGFSolver function call factorise, selected_inversion etc. class RGF gets created. Is this the best
 * way to handle things. Potentially merge them somehow? Maybe actually does not take any time.
 */
class RGFSolver: public Solver {

    private:

        int MPI_size;
        int MPI_rank;

        int threads_level1;

        /* matrix size */
        unsigned int n;                  /**< size of the matrix */
        unsigned int nnz;       /**< number of nonzeros */

        // to avoid redeclaration every time
        size_t i;

        // pardiso wants integer, RGF wants size_t, recast for now
        size_t ns_t;
        size_t nt_t;
        size_t nb_t;
        size_t no_t;

        SpMat Q;                /**< sparse precision matrix Q. Eigen format. */

        int* ia;                /**< CSR format. row indices. */
        int* ja;                /**< CSR format. col pointers. */
        double* a;              /**< CSR format. values. */

        double* b;              /**< right-hand side. */
        double* x;              /**< placeholder for solution. */
   	public:
   		RGFSolver(size_t ns_, size_t nt_, size_t nb_, size_t no_);

        /**
         * @brief not used for RGFSolver, only in PARDISO
         */
		void symbolic_factorization(SpMat& Q, int& init);

        /**
         * @brief numerical factorisation using block-wise factorisation on GPU. 
         * @param[in]       Q precision matrix to be factorised.
         * @param[inout]    log_det computes log determinant of Q.
         */
		void factorize(SpMat& Q, double& log_det);


        /**
         * @brief factorises and solves matrix in one call 
         * @param[in]       Q precision matrix.
         * @param[in]       rhs right-hand side of the system.
         * @param[inout]    sol solution of the system.
         * @param[inout]    log_det log determinant of Q.
         */ 
		void factorize_solve(SpMat& Q, Vect& rhs, Vect& sol, double &log_det);

        /**
         * @brief selected inversion of the diagonal elements of Q.
         * @param[in]       Q precision matrix.
         * @param[inout]    inv_diag inverse diagonal to hold the solution vector.
         * @note is there some way to potentially reuse Cholesky factor that is already on CPU?
         */
      	void selected_inversion(SpMat& Q, Vect& inv_diag);

      	// will also need a "simple inversion" method to independent of PARDISO. regular lapack should do (see pardiso)
        // OR not? Eigen function is probably fine, most likely also using lapack.

        /**
         * @brief class destructor. Frees memory allocated by RGF.
         */
      	~RGFSolver();

};


#endif