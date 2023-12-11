// BTASolver_dummy

#ifndef BTASOLVER_DUMMY_H
#define BTASOLVER_DUMMY_H

//#include "../BTA/BTA.H"

#include "Solver.h"

//extern "C" double 

 /**
 * @brief creates solver class using BTA-GPU for factorising, solving and selectively inverting linear system.
 * @details divided into set up, symbolic factorisation, numerical factorisation, numerical factorisation & solve 
 * and selected inversion (of the diagonal elements)
 * @note in each BTASolver function call factorise, selected_inversion etc. class BTA gets created. Is this the best
 * way to handle things. Potentially merge them somehow? Maybe actually does not take any time.
 */
class BTASolver: public Solver {

    private:

        /* matrix size */
        unsigned int n;                  /**< size of the matrix */
        unsigned int nnz;       /**< number of nonzeros */

        // to avoid redeclaration every time
        size_t i;

        // pardiso wants integer, BTA wants size_t, recast for now
        size_t ns_t;
        size_t nt_t;
        size_t nb_t;
        size_t no_t;

        //SpMat Q;                /**< sparse precision matrix Q. Eigen format. */

        //int* ia;                /**< CSR format. row indices. */
        //int* ja;                /**< CSR format. col pointers. */
        //double* a;              /**< CSR format. values. */

        double* b;              /**< right-hand side. */
        double* x;              /**< placeholder for solution. */

   	public:
        BTASolver(size_t ns_, size_t nt_, size_t nb_, size_t no_, int thread_ID_);

        /**
         * @brief not used for BTASolver, only in PARDISO
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
        void selected_inversion_diag(SpMat& Q, Vect& inv_diag);

        // function description TODO ... 
        void selected_inversion_diag_w_constr(SpMat& Q, const MatrixXd& D, Vect& inv_diag, MatrixXd& V);

        void selected_inversion_full(SpMat& Q, SpMat& Qinv);

        void selected_inversion_full_w_constr(SpMat& Q, const MatrixXd& D, SpMat& Qinv, MatrixXd& V);

        void compute_full_inverse(SpMat& Q, MatrixXd& Qinv);

        // will also need a "simple inversion" method to independent of PARDISO. regular lapack should do (see pardiso)
        // OR not? Eigen function is probably fine, most likely also using lapack.

        /**
         * @brief class destructor. Frees memory allocated by BTA.
         */
        ~BTASolver();

};


#endif
