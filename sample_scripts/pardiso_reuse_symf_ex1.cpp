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
    int* ia;
    int* ja;
    double* a;
    int nnz;

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

    // for now ... 
    double  b[8], x[8];


public:
    /** constructor pardiso init */
    /* do symbolic factorisation now or later? */
    PardisoSolver(int n_, int* ia_, int* ja_, double* a_) : n(n_), ia(ia_), ja(ja_), a(a_) {
    
        mtype = -2;

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

        msglvl = 1;         /* Print statistical information  */
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

        /* -------------------------------------------------------------------- */
        /* .. pardiso_printstats(...)                                           */
        /*    prints information on the matrix to STDOUT.                       */
        /*    Use this functionality only for debugging purposes                */
        /* -------------------------------------------------------------------- */

        pardiso_printstats (&mtype, &n, a, ia, ja, &nrhs, b, &error);
        if (error != 0) {
            printf("\nERROR right hand side: %d", error);
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
    void factorize(int* ia, int* ja, double* a, double& log_det){

        // TODO: check that pardisoSolver was initialised!

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
        printf("\nFactorization completed ...\n");

        log_det = dparm[32];

        /* -------------------------------------------------------------------- */    
        /* ..  Convert matrix back to 0-based C-notation.                       */
        /* -------------------------------------------------------------------- */ 
        
        for (i = 0; i < n+1; i++) {
            ia[i] -= 1;
        }
        for (i = 0; i < nnz; i++) {
            ja[i] -= 1;
        }

    }


    // numerical factorisation & solve
    void factorize_solve(int* ia, int* ja, double* a, double* b, double* x, double &log_det){

        // TODO: check that pardisoSolver was initialised!


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
        printf("\nFactorization completed ...\n");

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
        
        printf("\nSolve completed ... ");
        printf("\nThe solution of the system is: ");
        for (i = 0; i < n; i++) {
            printf("\n x [%d] = % f", i, x[i] );
        }
        printf ("\n\n");

        /* -------------------------------------------------------------------- */    
        /* ..  Convert matrix back to 0-based C-notation.                       */
        /* -------------------------------------------------------------------- */ 
    
        for (i = 0; i < n+1; i++) {
            ia[i] -= 1;
        }
        for (i = 0; i < nnz; i++) {
            ja[i] -= 1;
        }

    } // end factorise solve function


    void selected_inversion(int* ia, int* ja, double* a, double* inv_diag){

        // TODO: check that pardisoSolver was initialised!

        // can I somehow check if numerical factorisation was already done? For now just repeat.
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
        printf("\nFactorization completed ...\n");

        /* -------------------------------------------------------------------- */    
        /* ... Inverse factorization.                                           */                                       
        /* -------------------------------------------------------------------- */  


        printf("\nCompute Diagonal Elements of the inverse of A ... \n");
        phase = -22;
        iparm[35]  = 1; /*  no not overwrite internal factor L */ 

        pardiso (pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, b, x, &error,  dparm);

        /* print diagonal elements */
        for (k = 0; k < n; k++)
        {
            int j = ia[k]-1;
            printf ("Diagonal element of A^{-1} = %d %d %32.24e\n", k, ja[j]-1, a[j]);
        }

        /* -------------------------------------------------------------------- */    
        /* ..  Convert matrix back to 0-based C-notation.                       */
        /* -------------------------------------------------------------------- */ 
        for (i = 0; i < n+1; i++) {
            ia[i] -= 1;
        }
        for (i = 0; i < nnz; i++) {
            ja[i] -= 1;
        }

    } // end selected inversion function

    void release_memory(){
        /* -------------------------------------------------------------------- */    
        /* ..  Termination and release of memory.                               */
        /* -------------------------------------------------------------------- */    
        phase = -1;                 /* Release internal memory. */
        
        pardiso (pt, &maxfct, &mnum, &mtype, &phase,
                 &n, &ddum, ia, ja, &idum, &nrhs,
                 iparm, &msglvl, &ddum, &ddum, &error,  dparm);
    }

}; // end class


#if 0

int pardiso_set_up(int n, int* ia, int* ja, double* a){


    int      nnz = ia[n];
    int      mtype = -2;        /* Real symmetric matrix */

    /* RHS and solution vectors. */
    double   b[8], x[8];
    int      nrhs = 1;          /* Number of right hand sides. */

    /* Internal solver memory pointer pt,                  */
    /* 32-bit: int pt[64]; 64-bit: long int pt[64]         */
    /* or void *pt[64] should be OK on both architectures  */ 
    void    *pt[64]; 

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
         return 1; 
    }
    else
        printf("[PARDISO]: License check was successful ... \n");
    
    /* Numbers of processors, value of OMP_NUM_THREADS */
    var = getenv("OMP_NUM_THREADS");
    if(var != NULL)
        sscanf( var, "%d", &num_procs );
    else {
        printf("Set environment OMP_NUM_THREADS to 1");
        exit(1);
    }
    iparm[2]  = num_procs;

    maxfct = 1;		/* Maximum number of numerical factorizations.  */
    mnum   = 1;         /* Which factorization to use. */
    
    msglvl = 1;         /* Print statistical information  */
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

    /* Set right hand side to i. */
    for (i = 0; i < n; i++) {
        b[i] = i + 1;
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

/* -------------------------------------------------------------------- */
/* .. pardiso_printstats(...)                                           */
/*    prints information on the matrix to STDOUT.                       */
/*    Use this functionality only for debugging purposes                */
/* -------------------------------------------------------------------- */

    pardiso_printstats (&mtype, &n, a, ia, ja, &nrhs, b, &error);
    if (error != 0) {
        printf("\nERROR right hand side: %d", error);
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
    printf("\nNumber of factorization GFLOPS = %d", iparm[18]);

    return 0;

}


int pardiso_factorisation(int n, int* ia, int* ja, double* a, double& log_det){
   
/* -------------------------------------------------------------------- */
/* ..  Numerical factorization.                                         */
/* -------------------------------------------------------------------- */    

    /// NEW for LISA
    for (int loop = 0; loop < 1; loop++) {

    phase = 22;
    iparm[32] = 1; /* compute determinant */

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }
    printf("\nFactorization completed ...\n ");

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
    
    } // END LOOP LISA

    printf("\nSolve completed ... ");
    printf("\nThe solution of the system is: ");
    for (i = 0; i < n; i++) {
        printf("\n x [%d] = % f", i, x[i] );
    }
    printf ("\n\n");


/* -------------------------------------------------------------------- */    
/* ... Inverse factorization.                                           */                                       
/* -------------------------------------------------------------------- */  

    if (solver == 0)
    {
    	printf("\nCompute Diagonal Elements of the inverse of A ... \n");
	phase = -22;
        iparm[35]  = 1; /*  no not overwrite internal factor L */ 

        pardiso (pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, b, x, &error,  dparm);

       /* print diagonal elements */
       for (k = 0; k < n; k++)
       {
            int j = ia[k]-1;
            printf ("Diagonal element of A^{-1} = %d %d %32.24e\n", k, ja[j]-1, a[j]);
       }

    }   


/* -------------------------------------------------------------------- */    
/* ..  Convert matrix back to 0-based C-notation.                       */
/* -------------------------------------------------------------------- */ 
    for (i = 0; i < n+1; i++) {
        ia[i] -= 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] -= 1;
    }

/* -------------------------------------------------------------------- */    
/* ..  Termination and release of memory.                               */
/* -------------------------------------------------------------------- */    
    phase = -1;                 /* Release internal memory. */
    
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, &ddum, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);


    return 0;

} 

#endif

/* ===================================================================== */
int main( void ) 
{
    /* Matrix data. */
    int    n = 8;
    int    ia[ 9] = { 0, 4, 7, 9, 11, 14, 16, 17, 18 };
    
    std::cout << "ia : " << std::endl;
    for(int i=0; i<n; i++){
        std::cout << ia[i] << " ";
    }

    int    ja[18] = { 0,    2,       5, 6,
                         1, 2,    4,
                            2,             7,
                               3,       6,
                                  4, 5, 6,
                                     5,    7,
                                        6,
                                           7 };
    double  a[18] = { 7.0,      1.0,           2.0, 7.0,
                          -4.0, 8.0,           2.0,
                                1.0,                     5.0,
                                     7.0,           9.0,
                                          5.0, -1.0, 5.0,
                                               0.0,      5.0,
                                                   11.0,
                                                         5.0 };

    double* b;
    double* x;
    double* inv_diag;

    b        = new double [n];
    x        = new double [n]; 
    inv_diag = new double [n];


    /* Set right hand side to i. */
    for (int i = 0; i < n; i++) {
        b[i] = i + 1;
    } 

    double log_det;

    PardisoSolver* solver;
    solver = new PardisoSolver(n, ia, ja, a);
    solver->factorize(ia, ja, a, log_det); 
    std::cout << "\nlog det factorise       : " << log_det << std::endl;

    solver->factorize_solve(ia, ja, a, b, x, log_det);  

    std::cout << "\nlog det factorise solve : " << log_det << std::endl;
    std::cout << "x : ";
    for(int i=0; i<n; i++){
        std::cout << x[i] << " ";
    } 
    std::cout << "" << std::endl;

    solver->selected_inversion(ia, ja, a, inv_diag);

    solver->release_memory();



    return 0;
}

