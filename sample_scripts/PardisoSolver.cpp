/* -------------------------------------------------------------------- */
/*      Example program to show the use of the "PARDISO" routine        */
/*      on symmetric linear systems                                     */
/* -------------------------------------------------------------------- */
/*      This program can be downloaded from the following site:         */
/*      http://www.pardiso-project.org                                  */
/*                                                                      */
/*  (C) Olaf Schenk, Institute of Computational Science                 */
/*      Universita della Svizzera italiana, Lugano, Switzerland.        */
/*      Email: olaf.schenk@usi.ch                                       */
/* -------------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef Eigen::VectorXd Vector;
typedef Eigen::SparseMatrix<double> SpMat;


/* PARDISO prototype. */
extern "C" void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
extern "C" void pardiso     (void   *, int    *,   int *, int *,    int *, int *, 
                            double *, int    *,    int *, int *,   int *, int *,
                            int *, double *, double *, int *, double *);
extern "C" void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
extern "C" void pardiso_chkvec     (int *, int *, double *, int *);
extern "C" void pardiso_printstats (int *, int *, double *, int *, int *, int *,
                            double *, int *);


class PardisoSolver{

private:

    /* matrix size */
    int n;
    unsigned int nnz;

    SpMat Q;

    int* ia;
    int* ja;
    double* a;

    double* b;
    double* x;

    /* Internal solver memory pointer pt,                  */
    void *pt[64];

    /* Pardiso control parameters. */
    int      iparm[64];
    double   dparm[64];
    int      maxfct, mnum, phase, error, msglvl, solver;

    /* Number of processors. */
    int      num_procs;

    /* Auxiliary variables. */
    char    *var;
    int      i, k;

    double   ddum;              /* Double dummy */
    int      idum;              /* Integer dummy. */

    int     mtype;              /* matrix type */

    int     nrhs;



public:
    /** constructor pardiso init */
    /* do symbolic factorisation now or later? */
    PardisoSolver(SpMat& Q_) : Q(Q_) {

        n = Q.rows();

        // only take lower triangular part of A
        SpMat Q_lower = Q.triangularView<Lower>(); 

        // this time require CSR format

        nnz = Q_lower.nonZeros();
        std::cout << "number of non zeros : " << nnz << std::endl;

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

    
        mtype = -2;             /* set to positive semi-definite */

        nrhs = 1;               /* Number of right hand sides. */
        nnz = ia[n];

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
            printf("[PARDISO]: License check was successful ... \n");
        }

        /* Numbers of processors, value of OMP_NUM_THREADS */
        var = getenv("OMP_NUM_THREADS");
        if(var != NULL)
            sscanf( var, "%d", &num_procs );
        else {
            printf("Set environment OMP_NUM_THREADS to 1");
            exit(1);
        }

        iparm[2]  = num_procs;

        maxfct = 1;         /* Maximum number of numerical factorizations.  */
        mnum   = 1;         /* Which factorization to use. */

        msglvl = 0;         /* Print statistical information  */
        error  = 0;         /* Initialize error flag */

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

        printf("\nReordering completed ... ");
        printf("\nNumber of nonzeros in factors  = %d", iparm[17]);
        printf("\nNumber of factorization GFLOPS = %d\n", iparm[18]);

        /* -------------------------------------------------------------------- */    
        /* ..  Convert matrix back to 0-based C-notation.                       */
        /* -------------------------------------------------------------------- */ 
        
        for (i = 0; i < n+1; i++) {
            ia[i] -= 1;
        }
        for (i = 0; i < nnz; i++) {
            ja[i] -= 1;
        }


    } // end constructor


    /* ======================================================================== */

    // numerical factorisation
    void factorize(SpMat& Q, double& log_det){

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

        // is this a good idea?
        delete[] ia;
        delete[] ja;
        delete[] a;

    }

    // numerical factorisation & solve
    void factorize_solve(SpMat& Q, Vector& rhs, Vector& sol, double &log_det){

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

        // is this a good idea?
        delete[] ia;
        delete[] ja;
        delete[] a;

    } // end factorise solve function


    void selected_inversion(SpMat& Q, Vector& inv_diag){

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

        // is this a good idea?
        delete[] ia;
        delete[] ja;
        delete[] a;

    } // end selected inversion function

    void release_memory(){
        /* -------------------------------------------------------------------- */    
        /* ..  Termination and release of memory.                               */
        /* -------------------------------------------------------------------- */    
        phase = -1;                 /* Release internal memory. */
        
        pardiso (pt, &maxfct, &mnum, &mtype, &phase,
                 &n, &ddum, ia, ja, &idum, &nrhs,
                 iparm, &msglvl, &ddum, &ddum, &error,  dparm);

        // TODO: causes segmentation fault. why?1
        /*delete[] ia;
        delete[] ja;
        delete[] a;*/

    }

}; // end class