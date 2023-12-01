#ifndef EIGEN_CHOL_SOLVER_H
#define EIGEN_CHOL_SOLVER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iomanip>

#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
//#include <Eigen/CholmodSupport>

#include "Solver.h"

//#define PRINT_PAR
//#define PRINT_OMP

// get gflops manually from pardiso output
//#define MEAS_GFLOPS

//#define CHOLMOD

typedef Eigen::VectorXd Vect;
typedef Eigen::SparseMatrix<double> SpMat;

using namespace Eigen;

 /**
 * @brief creates solver class using pardiso for factorising, solving and selectively inverting linear system.
 * @details divided into set up, symbolic factorisation, numerical factorisation, numerical factorisation & solve 
 * and selected inversion (of the diagonal elements)
 */
class EigenCholSolver : public Solver{

private:


    int MPI_rank;           /**< pass on current MPI rank */

    int init;

    /* matrix size */
    int n;                  /**< size of the matrix */

    SpMat Q;                /**< sparse precision matrix Q. Eigen format. */

#ifdef CHOLMOD
    CholmodSupernodalLLT<SpMat> solverEigenQ;
#else
    //SimplicialLLT<SpMat> solverEigenQ;
    SimplicialLDLT<SpMat> solverEigenQ;
#endif


public:
     /**
     * @brief constructor. initialises parameters, check pardiso license.
     */
    EigenCholSolver(int& MPI_rank);


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
    void factorize(SpMat& Q, double& log_det, double& t_priorLatChol);

    void factorize_w_constr(SpMat& Q, const MatrixXd& D, double& log_det, MatrixXd& V);

    /**
     * @brief factorises and solves matrix in one call (to reuse pardiso objects)
     * @param[in]       Q precision matrix.
     * @param[in]       rhs right-hand side of the system.
     * @param[inout]    sol solution of the system.
     * @param[inout]    log_det log determinant of Q.
     */    
    void factorize_solve(SpMat& Q, Vect& rhs, Vect& sol, double &log_det, double& t_condLatChol, double& t_condLatSolve);

    void factorize_solve_w_constr(SpMat& Q, Vect& rhs, const MatrixXd& Dxy, double &log_det, Vect& sol, MatrixXd& V);
    
    /**
     * @brief selected inversion of the diagonal elements of Q.
     * @param[in]       Q precision matrix.
     * @param[inout]    inv_diag inverse diagonal to hold the solution vector.
     */
    void selected_inversion(SpMat& Q, Vect& inv_diag);

    void selected_inversion_w_constr(SpMat& Q, const MatrixXd& D, Vect& inv_diag, MatrixXd& V);


    /**
     * @brief inversion of the entire matrix (only meant for small matrices) by means of using identity 
     * right-hand side.
     * @param[in]       H dense matrix. 
     * @param[inout]    C inverse of H.
     */
    void compute_full_inverse(MatrixXd& H, MatrixXd& C);


     /**
     * @brief class destructor. Frees memory allocated by pardiso.
     */
    ~EigenCholSolver();

}; // end class


#endif


