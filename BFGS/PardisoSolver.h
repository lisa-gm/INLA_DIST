#ifndef PARDIDSO_SOLVER_H
#define PARDIDSO_SOLVER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "Solver.h"

//#define PRINT_PAR
//#define PRINT_OMP

typedef Eigen::VectorXd Vector;
typedef Eigen::SparseMatrix<double> SpMat;

using namespace Eigen;


/* PARDISO prototype. */
extern "C" void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
extern "C" void pardiso     (void   *, int    *,   int *, int *,    int *, int *, 
                            double *, int    *,    int *, int *,   int *, int *,
                            int *, double *, double *, int *, double *);
extern "C" void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
extern "C" void pardiso_chkvec     (int *, int *, double *, int *);
extern "C" void pardiso_printstats (int *, int *, double *, int *, int *, int *,
                            double *, int *);


 /**
 * @brief creates solver class using pardiso for factorising, solving and selectively inverting linear system.
 * @details divided into set up, symbolic factorisation, numerical factorisation, numerical factorisation & solve 
 * and selected inversion (of the diagonal elements)
 */
class PardisoSolver : public Solver{

private:

    /* matrix size */
    int n;                  /**< size of the matrix */
    unsigned int nnz;       /**< number of nonzeros */

    SpMat Q;                /**< sparse precision matrix Q. Eigen format. */

    int* ia;                /**< CSR format. row indices. */
    int* ja;                /**< CSR format. col pointers. */
    double* a;              /**< CSR format. values. */

    double* b;              /**< right-hand side. */
    double* x;              /**< placeholder for solution. */

    void *pt[64];           /**< Internal solver memory pointer pt */

    /* Pardiso control parameters. */
    int      iparm[64];
    double   dparm[64];
    int      maxfct, mnum, phase, error, msglvl, solver;

    int      num_procs;     /**< Number of processors. */

    /* Auxiliary variables. */
    int      i, k;

    double   ddum;              /**< Double dummy */
    int      idum;              /**< Integer dummy. */

    int     mtype;              /**< matrix type */
    int     nrhs;               /**< number of rhs. */
    int     init;               /**< flag that indicates if symbolic factorisation already performed. */



public:
     /**
     * @brief constructor. initialises parameters, check pardiso license.
     */
    PardisoSolver();


    /* ======================================================================== */
    /**
     * @brief performs the symbolic factorisation.
     * @param[in]       Q precision matrix to be factorised. 
     * @param[inout]    init integer value indicating status of symbolic factorisation. Changed to one at the end of the function.
     * @details For each PardisoSolver object symbolic factorisation only needs to be performed once as sparsity patters if
     * assumed to remain the same. 
     */
    void symbolic_factorization(SpMat& Q, int& init);


    /**
     * @brief numerical factorisation. 
     * @param[in]       Q precision matrix to be factorised.
     * @param[inout]    log_det computes log determinant of Q.
     */
    void factorize(SpMat& Q, double& log_det);


    /**
     * @brief factorises and solves matrix in one call (to reuse pardiso objects)
     * @param[in]       Q precision matrix.
     * @param[in]       rhs right-hand side of the system.
     * @param[inout]    sol solution of the system.
     * @param[inout]    log_det log determinant of Q.
     */    
    void factorize_solve(SpMat& Q, Vector& rhs, Vector& sol, double &log_det);


    /**
     * @brief selected inversion of the diagonal elements of Q.
     * @param[in]       Q precision matrix.
     * @param[inout]    inv_diag inverse diagonal to hold the solution vector.
     */
    void selected_inversion(SpMat& Q, Vector& inv_diag);

    /**
     * @brief inversion of the entire matrix (only meant for small matrices) by means of using identity 
     * right-hand side.
     * @param[in]       H dense matrix. 
     * @param[inout]    C inverse of H.
     */
    void compute_inverse_pardiso(MatrixXd& H, MatrixXd& C);


     /**
     * @brief class destructor. Frees memory allocated by pardiso.
     */
    ~PardisoSolver();

}; // end class


#endif
