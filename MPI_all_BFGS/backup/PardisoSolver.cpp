#if 1 

#include "PardisoSolver.h"

PardisoSolver::PardisoSolver(){

    mtype  = -2;             /* set to positive semi-definite */
    nrhs   = 1;              /* Number of right hand sides. */

    /* -------------------------------------------------------------------- */
    /* ..  Setup Pardiso control parameters.                                */
    /* -------------------------------------------------------------------- */

    error  = 0;
    solver = 0;              /* use sparse direct solver */
    pardisoinit (pt,  &mtype, &solver, iparm, dparm, &error); 

    if (error != 0) 
    {
    if (error == -10 )
       printf("No license file found \n");
    if (error == -11 )
       printf("License is expired \n");
    if (error == -12 )
       printf("Wrong username or hostname \n");
       exit(1); 
    } else {
        #ifdef PRINT_PAR
            printf("[PARDISO]: License check was successful ... \n");
        #endif
    }

    int threads_level2;

    #pragma omp parallel
    {
        threads_level2 = omp_get_max_threads();
    }

    #ifdef PRINT_OMP
        if(omp_get_thread_num() == 0){
            //char* var = getenv("OMP_NUM_THREADS");
            //std::cout << "OMP_NUM_THREADS = " << var << std::endl;
            std::cout << "Pardiso will be called with " << threads_level2 << " threads per solver. " << std::endl;
        }
        // printf("Thread rank: %d out of %d threads.\n", omp_get_thread_num(), omp_get_num_threads());
    #endif

    //iparm[2]  = num_procs;

    // make sure that this is called inside upper level parallel region 
    // to get number of threads on the second level 
    iparm[2] = threads_level2;

    maxfct = 1;         /* Maximum number of numerical factorizations.  */
    mnum   = 1;         /* Which factorization to use. */

    msglvl = 0;         /* Print statistical information  */
    error  = 0;         /* Initialize error flag */

    init = 0;           /* switch that determines if symbolic factorisation already happened */

} // end constructor


void PardisoSolver::symbolic_factorization(SpMat& Q, int& init){

    #ifdef PRINT_PAR
        std::cout << "in symbolic factorization." << std::endl;
    #endif

    n = Q.rows();

    // only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 

    // this time require CSR format

    nnz = Q_lower.nonZeros();

    #ifdef PRINT_PAR
        std::cout << "number of non zeros : " << nnz << std::endl;
    #endif

    int* ia; 
    int* ja;
    double* a; 

    // allocate memory
    ia = new int [n+1];
    ja = new int [nnz];
    a = new double [nnz];

    Q_lower.makeCompressed();

    for (int i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
    }  

    for (int i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
    }  

    for (int i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
    } 

    nnz = ia[n];

    /* -------------------------------------------------------------------- */
    /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
    /*     notation.                                                        */
    /* -------------------------------------------------------------------- */
    for (i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] += 1;
    }

    /* -------------------------------------------------------------------- */
    /*  .. pardiso_chk_matrix(...)                                          */
    /*     Checks the consistency of the given matrix.                      */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);
    exit(1);
    }

    /* -------------------------------------------------------------------- */
    /* ..  Reordering and Symbolic Factorization.  This step also allocates */
    /*     all memory that is necessary for the factorization.              */
    /* -------------------------------------------------------------------- */
    phase = 11; 

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
     &n, a, ia, ja, &idum, &nrhs,
         iparm, &msglvl, &ddum, &ddum, &error, dparm);

    if (error != 0) {
        printf("\nERROR during symbolic factorization: %d", error);
        exit(1);
    }

    #ifdef PRINT_PAR
        printf("\nReordering completed ... ");
        printf("\nNumber of nonzeros in factors  = %d", iparm[17]);
        printf("\nNumber of factorization GFLOPS = %d\n", iparm[18]);
    #endif

    // set init to 1 to indicate that symbolic factorisation happened
    init = 1;

    delete[] ia;
    delete[] ja;
    delete[] a;

}


void PardisoSolver::factorize(SpMat& Q, double& log_det){

    #ifdef PRINT_PAR
        std::cout << "init = " << init << std::endl;
    #endif

    if(init == 0){
        symbolic_factorization(Q, init);
    }

    // check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n = %d.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

    // only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 

    // check if nnz and Q_lower.nonZeros match
    if(nnz != Q_lower.nonZeros()){
        printf("Initial number of nonzeros and current number of nonzeros don't match!\n");
        printf("nnz = %d.\n nnz(Q_lower) = %ld\n", nnz, Q_lower.nonZeros());
    }

    int* ia; 
    int* ja;
    double* a; 

    // allocate memory
    ia = new int [n+1];
    ja = new int [nnz];
    a = new double [nnz];

    Q_lower.makeCompressed();

    for (int i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
    }  

    for (int i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
    }  

    for (int i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
    }

    // TODO: save work, some already 1-based ... make sure that this is bullet proof.
    /* -------------------------------------------------------------------- */
    /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
    /*     notation.                                                        */
    /* -------------------------------------------------------------------- */
    
    for (i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] += 1;
    }      

    /* -------------------------------------------------------------------- */
    /*  .. pardiso_chk_matrix(...)                                          */
    /*     Checks the consistency of the given matrix.                      */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);
    exit(1);
    }

    phase = 22;
    iparm[32] = 1; /* compute determinant */

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }
    //printf("\nFactorization completed ...\n");

    log_det = dparm[32];

    delete[] ia;
    delete[] ja;
    delete[] a;

}

 
void PardisoSolver::factorize_solve(SpMat& Q, Vector& rhs, Vector& sol, double &log_det){

    #ifdef PRINT_PAR
        std::cout << "init = " << init << std::endl;
    #endif

    if(init == 0){
        symbolic_factorization(Q, init);
    }

// check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n = %d.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

    // only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 

    // check if nnz and Q_lower.nonZeros match
    if(nnz != Q_lower.nonZeros()){
        printf("Initial number of nonzeros and current number of nonzeros don't match!\n");
        printf("nnz = %d.\n nnz(Q_lower) = %ld\n", nnz, Q_lower.nonZeros());
    }

    int* ia; 
    int* ja;
    double* a; 

    // allocate memory
    ia = new int [n+1];
    ja = new int [nnz];
    a = new double [nnz];

    Q_lower.makeCompressed();

    for (int i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
    }  

    for (int i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
    }  

    for (int i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
    }

    b = new double [n];
    x = new double [n];

    /* Set right hand side to i. */
    for (int i = 0; i < n; i++) {
        b[i] = rhs[i];
    } 


    /* -------------------------------------------------------------------- */
    /* ..  pardiso_chkvec(...)                                              */
    /*     Checks the given vectors for infinite and NaN values             */
    /*     Input parameters (see PARDISO user manual for a description):    */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkvec (&n, &nrhs, b, &error);
    if (error != 0) {
        printf("\nERROR  in right hand side: %d", error);
        exit(1);
    }


    // TODO: save work, some already 1-based ... make sure that this is bullet proof.
    /* -------------------------------------------------------------------- */
    /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
    /*     notation.                                                        */
    /* -------------------------------------------------------------------- */
    
    for (i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] += 1;
    }      

    /* -------------------------------------------------------------------- */
    /*  .. pardiso_chk_matrix(...)                                          */
    /*     Checks the consistency of the given matrix.                      */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);
    exit(1);
    }

    phase = 22;
    iparm[32] = 1; /* compute determinant */

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }
    //printf("\nFactorization completed ...\n");

    log_det = dparm[32];

    /* -------------------------------------------------------------------- */    
    /* ..  Back substitution and iterative refinement.                      */
    /* -------------------------------------------------------------------- */    
    phase = 33;

    iparm[7] = 1;       /* Max numbers of iterative refinement steps. */
   
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, b, x, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during solution: %d", error);
        exit(3);
    }
    
    //printf("\nSolve completed ... ");
    for (i = 0; i < n; i++) {
        //printf("\n x [%d] = % f", i, x[i] );
        sol(i) = x[i];
    }

    delete[] ia;
    delete[] ja;
    delete[] a;

    delete[] x;
    delete[] b;

} // end factorise solve function


void PardisoSolver::selected_inversion(SpMat& Q, Vector& inv_diag){

    #ifdef PRINT_PAR
        std::cout << "init = " << init << std::endl;
    #endif

    if(init == 0){
        symbolic_factorization(Q, init);
    }

    // check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n = %d.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

    // only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 

    // check if nnz and Q_lower.nonZeros match
    if(nnz != Q_lower.nonZeros()){
        printf("Initial number of nonzeros and current number of nonzeros don't match!\n");
        printf("nnz = %d.\n nnz(Q_lower) = %ld\n", nnz, Q_lower.nonZeros());
    }

    int* ia; 
    int* ja;
    double* a; 

    // allocate memory
    ia = new int [n+1];
    ja = new int [nnz];
    a = new double [nnz];

    Q_lower.makeCompressed();

    for (int i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
    }  

    for (int i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
    }  

    for (int i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
    }

    // TODO: make already one-based in the above loop
    /* -------------------------------------------------------------------- */
    /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
    /*     notation.                                                        */
    /* -------------------------------------------------------------------- */
    
    for (i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] += 1;
    }      

    /* -------------------------------------------------------------------- */
    /*  .. pardiso_chk_matrix(...)                                          */
    /*     Checks the consistency of the given matrix.                      */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);
    exit(1);
    }

    phase = 22;
    iparm[32] = 1; /* compute determinant */

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }
    //printf("\nFactorization completed ...\n");

    /* -------------------------------------------------------------------- */    
    /* ... Inverse factorization.                                           */                                       
    /* -------------------------------------------------------------------- */  

    // what do we need both b & x for here?
    b = new double [n];
    x = new double [n];

    //printf("\nCompute Diagonal Elements of the inverse of A ... \n");
    phase = -22;
    iparm[35]  = 1; /*  no not overwrite internal factor L */ 

    pardiso (pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
         iparm, &msglvl, b, x, &error,  dparm);

    /* print diagonal elements */
    for (k = 0; k < n; k++)
    {
        int j = ia[k]-1;
        //printf ("Diagonal element of A^{-1} = %d %d %32.24e\n", k, ja[j]-1, a[j]);
        inv_diag(k) = a[j];
    }

    delete[] ia;
    delete[] ja;
    delete[] a;

    delete[] x;
    delete[] b;

} // end selected inversion function

// function to completely invert small n x n matrix using identity as rhs
// assumes function to be SYMMETRIC (which is the case for hessian)
void PardisoSolver::compute_inverse_pardiso(MatrixXd& Q, MatrixXd& C){

    // convert to sparse matrix for pardiso
    SpMat H = Q.sparseView();

    #ifdef PRINT_PAR
        std::cout << "init = " << init << std::endl;
    #endif

    if(init == 0){
        symbolic_factorization(H, init);
    }

// check if n and H.size() match
    if(n != H.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n = %d.\nnrows(H) = %ld.\n", n, H.rows());
        exit(1);
    }

    // only take lower triangular part of A
    SpMat H_lower = H.triangularView<Lower>(); 

    // check if nnz and H_lower.nonZeros match
    if(nnz != H_lower.nonZeros()){
        printf("Initial number of nonzeros and current number of nonzeros don't match!\n");
        printf("nnz = %d.\n nnz(H_lower) = %ld\n", nnz, H_lower.nonZeros());
    }

    // IMPORTANT : change the number of right-hand sides
    nrhs = n;

    int* ia; 
    int* ja;
    double* a; 

    // allocate memory
    ia = new int [n+1];
    ja = new int [nnz];
    a = new double [nnz];

    H_lower.makeCompressed();

    for (int i = 0; i < n+1; ++i){
        ia[i] = H_lower.outerIndexPtr()[i]; 
    }  

    for (int i = 0; i < nnz; ++i){
        ja[i] = H_lower.innerIndexPtr()[i];
    }  

    for (int i = 0; i < nnz; ++i){
        a[i] = H_lower.valuePtr()[i];
    }

    int n2 = n*n;
    b = new double [n2];
    x = new double [n2];

    /* Set right hand side to Identity. */
    for (int i = 0; i < n2; i++) {
        if(i % (n+1) == 0){
            b[i] = 1.0;
        } else{
            b[i] = 0;
        }
    } 

    /* -------------------------------------------------------------------- */
    /* ..  pardiso_chkvec(...)                                              */
    /*     Checks the given vectors for infinite and NaN values             */
    /*     Input parameters (see PARDISO user manual for a description):    */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkvec (&n, &nrhs, b, &error);
    if (error != 0) {
        printf("\nERROR  in right hand side: %d", error);
        exit(1);
    }


    // TODO: save work, some already 1-based ... make sure that this is bullet proof.
    /* -------------------------------------------------------------------- */
    /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
    /*     notation.                                                        */
    /* -------------------------------------------------------------------- */
    
    for (i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] += 1;
    }      

    /* -------------------------------------------------------------------- */
    /*  .. pardiso_chk_matrix(...)                                          */
    /*     Checks the consistency of the given matrix.                      */
    /*     Use this functionality only for debugging purposes               */
    /* -------------------------------------------------------------------- */

    pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);
    exit(1);
    }

    phase = 22;

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }
    //printf("\nFactorization completed ...\n");

    /* -------------------------------------------------------------------- */    
    /* ..  Back substitution and iterative refinement.                      */
    /* -------------------------------------------------------------------- */    
    phase = 33;

    iparm[7] = 1;       /* Max numbers of iterative refinement steps. */
   
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, b, x, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during solution: %d", error);
        exit(3);
    }
    
    //printf("\nSolve completed ... ");
    /*for (i = 0; i < n2; i++) {
        printf("\n x [%d] = % f", i, x[i] );
        //sol(i) = x[i];
    }*/

    // fill C with x values
    C = MatrixXd::Map(x, n, n);

    //std::cout << "\nC = \n" << C << std::endl;


    delete[] ia;
    delete[] ja;
    delete[] a;

    delete[] x;
    delete[] b;


}


PardisoSolver::~PardisoSolver(){
    /* -------------------------------------------------------------------- */    
    /* ..  Termination and release of memory.                               */
    /* -------------------------------------------------------------------- */    
    phase = -1;                 /* Release internal memory. */
    
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, &ddum, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);

}

#endif


#if 0 

#ifndef __PARDIDSO_SOLVER
#define __PARDIDSO_SOLVER

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

//#define PRINT_PAR
//#define PRINT_OMP

using Eigen::VectorXd;
using Eigen::MatrixXd;

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
class PardisoSolver{

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
    PardisoSolver(){

        mtype = -2;             /* set to positive semi-definite */
        nrhs = 1;               /* Number of right hand sides. */

        /* -------------------------------------------------------------------- */
        /* ..  Setup Pardiso control parameters.                                */
        /* -------------------------------------------------------------------- */

        error = 0;
        solver=0;/* use sparse direct solver */
        pardisoinit (pt,  &mtype, &solver, iparm, dparm, &error); 

        if (error != 0) 
        {
        if (error == -10 )
           printf("No license file found \n");
        if (error == -11 )
           printf("License is expired \n");
        if (error == -12 )
           printf("Wrong username or hostname \n");
           exit(1); 
        } else {
            #ifdef PRINT_PAR
                printf("[PARDISO]: License check was successful ... \n");
            #endif
        }

        int threads_level2;

        #pragma omp parallel
        {
            threads_level2 = omp_get_max_threads();
        }

        #ifdef PRINT_OMP
            if(omp_get_thread_num() == 0){
                //char* var = getenv("OMP_NUM_THREADS");
                //std::cout << "OMP_NUM_THREADS = " << var << std::endl;
                std::cout << "Pardiso will be called with " << threads_level2 << " threads per solver. " << std::endl;
            }
            // printf("Thread rank: %d out of %d threads.\n", omp_get_thread_num(), omp_get_num_threads());
        #endif

        //iparm[2]  = num_procs;

        // make sure that this is called inside upper level parallel region 
        // to get number of threads on the second level 
        iparm[2] = threads_level2;

        maxfct = 1;         /* Maximum number of numerical factorizations.  */
        mnum   = 1;         /* Which factorization to use. */

        msglvl = 0;         /* Print statistical information  */
        error  = 0;         /* Initialize error flag */

        init = 0;           /* switch that determines if symbolic factorisation already happened */

    } // end constructor


    /* ======================================================================== */
    /**
     * @brief performs the symbolic factorisation.
     * @param[in]       Q precision matrix to be factorised. 
     * @param[inout]    init integer value indicating status of symbolic factorisation. Changed to one at the end of the function.
     * @details For each PardisoSolver object symbolic factorisation only needs to be performed once as sparsity patters if
     * assumed to remain the same. 
     */
    void symbolic_factorization(SpMat& Q, int& init){

        #ifdef PRINT_PAR
            std::cout << "in symbolic factorization." << std::endl;
        #endif

        n = Q.rows();

        // only take lower triangular part of A
        SpMat Q_lower = Q.triangularView<Lower>(); 

        // this time require CSR format

        nnz = Q_lower.nonZeros();

        #ifdef PRINT_PAR
            std::cout << "number of non zeros : " << nnz << std::endl;
        #endif

        int* ia; 
        int* ja;
        double* a; 

        // allocate memory
        ia = new int [n+1];
        ja = new int [nnz];
        a = new double [nnz];

        Q_lower.makeCompressed();

        for (int i = 0; i < n+1; ++i){
            ia[i] = Q_lower.outerIndexPtr()[i]; 
        }  

        for (int i = 0; i < nnz; ++i){
            ja[i] = Q_lower.innerIndexPtr()[i];
        }  

        for (int i = 0; i < nnz; ++i){
            a[i] = Q_lower.valuePtr()[i];
        } 

        nnz = ia[n];

        /* -------------------------------------------------------------------- */
        /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
        /*     notation.                                                        */
        /* -------------------------------------------------------------------- */
        for (i = 0; i < n+1; i++) {
            ia[i] += 1;
        }
        for (i = 0; i < nnz; i++) {
            ja[i] += 1;
        }

        /* -------------------------------------------------------------------- */
        /*  .. pardiso_chk_matrix(...)                                          */
        /*     Checks the consistency of the given matrix.                      */
        /*     Use this functionality only for debugging purposes               */
        /* -------------------------------------------------------------------- */

        pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
        if (error != 0) {
            printf("\nERROR in consistency of matrix: %d", error);
        exit(1);
        }

        /* -------------------------------------------------------------------- */
        /* ..  Reordering and Symbolic Factorization.  This step also allocates */
        /*     all memory that is necessary for the factorization.              */
        /* -------------------------------------------------------------------- */
        phase = 11; 

        pardiso (pt, &maxfct, &mnum, &mtype, &phase,
         &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error, dparm);

        if (error != 0) {
            printf("\nERROR during symbolic factorization: %d", error);
            exit(1);
        }

        #ifdef PRINT_PAR
            printf("\nReordering completed ... ");
            printf("\nNumber of nonzeros in factors  = %d", iparm[17]);
            printf("\nNumber of factorization GFLOPS = %d\n", iparm[18]);
        #endif

        // set init to 1 to indicate that symbolic factorisation happened
        init = 1;

        delete[] ia;
        delete[] ja;
        delete[] a;

    }

    /**
     * @brief numerical factorisation. 
     * @param[in]       Q precision matrix to be factorised.
     * @param[inout]    log_det computes log determinant of Q.
     */
    void factorize(SpMat& Q, double& log_det){

        #ifdef PRINT_PAR
            std::cout << "init = " << init << std::endl;
        #endif

        if(init == 0){
            symbolic_factorization(Q, init);
        }

        // check if n and Q.size() match
        if(n != Q.rows()){
            printf("\nInitialised matrix size and current matrix size don't match!\n");
            printf("n = %d.\nnrows(Q) = %ld.\n", n, Q.rows());
            exit(1);
        }

        // only take lower triangular part of A
        SpMat Q_lower = Q.triangularView<Lower>(); 

        // check if nnz and Q_lower.nonZeros match
        if(nnz != Q_lower.nonZeros()){
            printf("Initial number of nonzeros and current number of nonzeros don't match!\n");
            printf("nnz = %d.\n nnz(Q_lower) = %ld\n", nnz, Q_lower.nonZeros());
        }

        int* ia; 
        int* ja;
        double* a; 

        // allocate memory
        ia = new int [n+1];
        ja = new int [nnz];
        a = new double [nnz];

        Q_lower.makeCompressed();

        for (int i = 0; i < n+1; ++i){
            ia[i] = Q_lower.outerIndexPtr()[i]; 
        }  

        for (int i = 0; i < nnz; ++i){
            ja[i] = Q_lower.innerIndexPtr()[i];
        }  

        for (int i = 0; i < nnz; ++i){
            a[i] = Q_lower.valuePtr()[i];
        }

        // TODO: save work, some already 1-based ... make sure that this is bullet proof.
        /* -------------------------------------------------------------------- */
        /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
        /*     notation.                                                        */
        /* -------------------------------------------------------------------- */
        
        for (i = 0; i < n+1; i++) {
            ia[i] += 1;
        }
        for (i = 0; i < nnz; i++) {
            ja[i] += 1;
        }      

        /* -------------------------------------------------------------------- */
        /*  .. pardiso_chk_matrix(...)                                          */
        /*     Checks the consistency of the given matrix.                      */
        /*     Use this functionality only for debugging purposes               */
        /* -------------------------------------------------------------------- */

        pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
        if (error != 0) {
            printf("\nERROR in consistency of matrix: %d", error);
        exit(1);
        }

        phase = 22;
        iparm[32] = 1; /* compute determinant */

        pardiso (pt, &maxfct, &mnum, &mtype, &phase,
                 &n, a, ia, ja, &idum, &nrhs,
                 iparm, &msglvl, &ddum, &ddum, &error,  dparm);
       
        if (error != 0) {
            printf("\nERROR during numerical factorization: %d", error);
            exit(2);
        }
        //printf("\nFactorization completed ...\n");

        log_det = dparm[32];

        delete[] ia;
        delete[] ja;
        delete[] a;

    }

    /**
     * @brief factorises and solves matrix in one call (to reuse pardiso objects)
     * @param[in]       Q precision matrix.
     * @param[in]       rhs right-hand side of the system.
     * @param[inout]    sol solution of the system.
     * @param[inout]    log_det log determinant of Q.
     */    
    void factorize_solve(SpMat& Q, Vector& rhs, Vector& sol, double &log_det){

        #ifdef PRINT_PAR
            std::cout << "init = " << init << std::endl;
        #endif

        if(init == 0){
            symbolic_factorization(Q, init);
        }

    // check if n and Q.size() match
        if(n != Q.rows()){
            printf("\nInitialised matrix size and current matrix size don't match!\n");
            printf("n = %d.\nnrows(Q) = %ld.\n", n, Q.rows());
            exit(1);
        }

        // only take lower triangular part of A
        SpMat Q_lower = Q.triangularView<Lower>(); 

        // check if nnz and Q_lower.nonZeros match
        if(nnz != Q_lower.nonZeros()){
            printf("Initial number of nonzeros and current number of nonzeros don't match!\n");
            printf("nnz = %d.\n nnz(Q_lower) = %ld\n", nnz, Q_lower.nonZeros());
        }

        int* ia; 
        int* ja;
        double* a; 

        // allocate memory
        ia = new int [n+1];
        ja = new int [nnz];
        a = new double [nnz];

        Q_lower.makeCompressed();

        for (int i = 0; i < n+1; ++i){
            ia[i] = Q_lower.outerIndexPtr()[i]; 
        }  

        for (int i = 0; i < nnz; ++i){
            ja[i] = Q_lower.innerIndexPtr()[i];
        }  

        for (int i = 0; i < nnz; ++i){
            a[i] = Q_lower.valuePtr()[i];
        }

        b = new double [n];
        x = new double [n];

        /* Set right hand side to i. */
        for (int i = 0; i < n; i++) {
            b[i] = rhs[i];
        } 


        /* -------------------------------------------------------------------- */
        /* ..  pardiso_chkvec(...)                                              */
        /*     Checks the given vectors for infinite and NaN values             */
        /*     Input parameters (see PARDISO user manual for a description):    */
        /*     Use this functionality only for debugging purposes               */
        /* -------------------------------------------------------------------- */

        pardiso_chkvec (&n, &nrhs, b, &error);
        if (error != 0) {
            printf("\nERROR  in right hand side: %d", error);
            exit(1);
        }


        // TODO: save work, some already 1-based ... make sure that this is bullet proof.
        /* -------------------------------------------------------------------- */
        /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
        /*     notation.                                                        */
        /* -------------------------------------------------------------------- */
        
        for (i = 0; i < n+1; i++) {
            ia[i] += 1;
        }
        for (i = 0; i < nnz; i++) {
            ja[i] += 1;
        }      

        /* -------------------------------------------------------------------- */
        /*  .. pardiso_chk_matrix(...)                                          */
        /*     Checks the consistency of the given matrix.                      */
        /*     Use this functionality only for debugging purposes               */
        /* -------------------------------------------------------------------- */

        pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
        if (error != 0) {
            printf("\nERROR in consistency of matrix: %d", error);
        exit(1);
        }

        phase = 22;
        iparm[32] = 1; /* compute determinant */

        pardiso (pt, &maxfct, &mnum, &mtype, &phase,
                 &n, a, ia, ja, &idum, &nrhs,
                 iparm, &msglvl, &ddum, &ddum, &error,  dparm);
       
        if (error != 0) {
            printf("\nERROR during numerical factorization: %d", error);
            exit(2);
        }
        //printf("\nFactorization completed ...\n");

        log_det = dparm[32];

        /* -------------------------------------------------------------------- */    
        /* ..  Back substitution and iterative refinement.                      */
        /* -------------------------------------------------------------------- */    
        phase = 33;

        iparm[7] = 1;       /* Max numbers of iterative refinement steps. */
       
        pardiso (pt, &maxfct, &mnum, &mtype, &phase,
                 &n, a, ia, ja, &idum, &nrhs,
                 iparm, &msglvl, b, x, &error,  dparm);
       
        if (error != 0) {
            printf("\nERROR during solution: %d", error);
            exit(3);
        }
        
        //printf("\nSolve completed ... ");
        for (i = 0; i < n; i++) {
            //printf("\n x [%d] = % f", i, x[i] );
            sol(i) = x[i];
        }

        delete[] ia;
        delete[] ja;
        delete[] a;

        delete[] x;
        delete[] b;

    } // end factorise solve function

    /**
     * @brief selected inversion of the diagonal elements of Q.
     * @param[in]       Q precision matrix.
     * @param[inout]    inv_diag inverse diagonal to hold the solution vector.
     */
    void selected_inversion(SpMat& Q, Vector& inv_diag){

        #ifdef PRINT_PAR
            std::cout << "init = " << init << std::endl;
        #endif

        if(init == 0){
            symbolic_factorization(Q, init);
        }

        // check if n and Q.size() match
        if(n != Q.rows()){
            printf("\nInitialised matrix size and current matrix size don't match!\n");
            printf("n = %d.\nnrows(Q) = %ld.\n", n, Q.rows());
            exit(1);
        }

        // only take lower triangular part of A
        SpMat Q_lower = Q.triangularView<Lower>(); 

        // check if nnz and Q_lower.nonZeros match
        if(nnz != Q_lower.nonZeros()){
            printf("Initial number of nonzeros and current number of nonzeros don't match!\n");
            printf("nnz = %d.\n nnz(Q_lower) = %ld\n", nnz, Q_lower.nonZeros());
        }

        int* ia; 
        int* ja;
        double* a; 

        // allocate memory
        ia = new int [n+1];
        ja = new int [nnz];
        a = new double [nnz];

        Q_lower.makeCompressed();

        for (int i = 0; i < n+1; ++i){
            ia[i] = Q_lower.outerIndexPtr()[i]; 
        }  

        for (int i = 0; i < nnz; ++i){
            ja[i] = Q_lower.innerIndexPtr()[i];
        }  

        for (int i = 0; i < nnz; ++i){
            a[i] = Q_lower.valuePtr()[i];
        }

        // TODO: make already one-based in the above loop
        /* -------------------------------------------------------------------- */
        /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
        /*     notation.                                                        */
        /* -------------------------------------------------------------------- */
        
        for (i = 0; i < n+1; i++) {
            ia[i] += 1;
        }
        for (i = 0; i < nnz; i++) {
            ja[i] += 1;
        }      

        /* -------------------------------------------------------------------- */
        /*  .. pardiso_chk_matrix(...)                                          */
        /*     Checks the consistency of the given matrix.                      */
        /*     Use this functionality only for debugging purposes               */
        /* -------------------------------------------------------------------- */

        pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
        if (error != 0) {
            printf("\nERROR in consistency of matrix: %d", error);
        exit(1);
        }

        phase = 22;
        iparm[32] = 1; /* compute determinant */

        pardiso (pt, &maxfct, &mnum, &mtype, &phase,
                 &n, a, ia, ja, &idum, &nrhs,
                 iparm, &msglvl, &ddum, &ddum, &error,  dparm);
       
        if (error != 0) {
            printf("\nERROR during numerical factorization: %d", error);
            exit(2);
        }
        //printf("\nFactorization completed ...\n");

        /* -------------------------------------------------------------------- */    
        /* ... Inverse factorization.                                           */                                       
        /* -------------------------------------------------------------------- */  

        // what do we need both b & x for here?
        b = new double [n];
        x = new double [n];

        //printf("\nCompute Diagonal Elements of the inverse of A ... \n");
        phase = -22;
        iparm[35]  = 1; /*  no not overwrite internal factor L */ 

        pardiso (pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, b, x, &error,  dparm);

        /* print diagonal elements */
        for (k = 0; k < n; k++)
        {
            int j = ia[k]-1;
            //printf ("Diagonal element of A^{-1} = %d %d %32.24e\n", k, ja[j]-1, a[j]);
            inv_diag(k) = a[j];
        }

        delete[] ia;
        delete[] ja;
        delete[] a;

        delete[] x;
        delete[] b;

    } // end selected inversion function

     /**
     * @brief class destructor. Frees memory allocated by pardiso.
     */
    ~PardisoSolver(){
        /* -------------------------------------------------------------------- */    
        /* ..  Termination and release of memory.                               */
        /* -------------------------------------------------------------------- */    
        phase = -1;                 /* Release internal memory. */
        
        pardiso (pt, &maxfct, &mnum, &mtype, &phase,
                 &n, &ddum, ia, ja, &idum, &nrhs,
                 iparm, &msglvl, &ddum, &ddum, &error,  dparm);

    }

}; // end class


#endif

#endif