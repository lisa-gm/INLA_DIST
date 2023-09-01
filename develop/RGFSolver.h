#ifndef RGFSOLVER_H
#define RGFSOLVER_H

#include "mpi.h"

#include "Solver.h"
#include "../RGF/RGF.H"
#include "helper_functions.h"
//#include "RGF.H"

//#define PRINT_MSG
//#define PRINT_TIMES
//#define GFLOPS

/*
#if 0
typedef CPX T;
#define assign_T(val) CPX(val, 0.0)
#else
typedef double T;
#define assign_T(val) val
#endif
*/


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

        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;

        int threads_level1;
        int thread_ID;

        /* matrix size */
        unsigned int nnz;       /**< number of nonzeros */

        // to avoid redeclaration every time
        size_t i;

        int GPU_rank;
        // pardiso wants integer, RGF wants size_t, recast for now
        size_t ns_t;
        size_t nt_t;
        size_t nb_t;
        size_t no_t;

        size_t n;

        SpMat Q;                /**< sparse precision matrix Q. Eigen format. */

        int* ia;                /**< CSR format. row indices. */
        int* ja;                /**< CSR format. col pointers. */
        double* a;              /**< CSR format. values. */

        double* b;              /**< right-hand side. */
        double* x;              /**< placeholder for solution. */

        RGF<double> *solver;    /**< RGF solver object */

   	public:
   		RGFSolver(size_t ns_, size_t nt_, size_t nb_, size_t no_, int thread_ID_);

        /**
         * @brief not used for RGFSolver, only in PARDISO
         */
		void symbolic_factorization(SpMat& Q, int& init);

        /**
         * @brief numerical factorisation using block-wise factorisation on GPU. 
         * @param[in]       Q precision matrix to be factorised.
         * @param[inout]    log_det computes log determinant of Q.
         */
		void factorize(SpMat& Q, double& log_det, double& t_priorLatChol);

        // function description TODO ...
        void factorize_w_constr(SpMat& Q, const MatrixXd& D, double& log_det, MatrixXd& V);

        /**
         * @brief factorises and solves matrix in one call 
         * @param[in]       Q precision matrix.
         * @param[in]       rhs right-hand side of the system.
         * @param[inout]    sol solution of the system.
         * @param[inout]    log_det log determinant of Q.
         */ 
		void factorize_solve(SpMat& Q, Vect& rhs, Vect& sol, double &log_det, double& t_condLatChol, double& t_condLatSolve);

        // function description TODO ...
        void factorize_solve_w_constr(SpMat& Q, Vect& rhs, const MatrixXd& Dxy, double &log_det, Vect& sol, MatrixXd& V);

        /**
         * @brief selected inversion of the diagonal elements of Q.
         * @param[in]       Q precision matrix.
         * @param[inout]    inv_diag inverse diagonal to hold the solution vector.
         * @note is there some way to potentially reuse Cholesky factor that is already on CPU?
         */
      	void selected_inversion(SpMat& Q, Vect& inv_diag);

        // function description TODO ... 
        void selected_inversion_w_constr(SpMat& Q, const MatrixXd& D, Vect& inv_diag, MatrixXd& V);

      	// will also need a "simple inversion" method to independent of PARDISO. regular lapack should do (see pardiso)
        // OR not? Eigen function is probably fine, most likely also using lapack.

        /**
         * @brief class destructor. Frees memory allocated by RGF.
         */
      	~RGFSolver();

};


#endif
