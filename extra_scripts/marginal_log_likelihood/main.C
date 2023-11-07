#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/SparseExtra>   // includes saveMarket

#include <armadillo>
#include "generate_testMat_st_s_field.cpp"
#include "../../read_write_functions.cpp"

using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef Eigen::VectorXd Vect;

//#define PRINT_MSG

#if 0
typedef CPX T;
#define assign_T(val) CPX(val, 0.0)
#else
typedef double T;
#define assign_T(val) val
#endif

/* PARDISO prototype. */
extern "C" void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
extern "C" void pardiso     (void   *, int    *,   int *, int *,    int *, int *, 
                  double *, int    *,    int *, int *,   int *, int *,
                     int *, double *, double *, int *, double *);
extern "C" void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
extern "C" void pardiso_chkvec     (int *, int *, double *, int *);
extern "C" void pardiso_printstats (int *, int *, double *, int *, int *, int *,
                           double *, int *);


void construct_Q_spatial(SpMat& Qs, Vect theta, SpMat& c0, SpMat& g1, SpMat& g2){

	// Qs <- g[1]^2*Qgk.fun(sfem, g[2], order)
	// return(g^4 * fem$c0 + 2 * g^2 * fem$g1 + fem$g2)
	double exp_theta1 = exp(theta[0]);
	double exp_theta2 = exp(theta[1]);
	//double exp_theta1 = -3;
	//double exp_theta2 = 1.5;

	Qs = pow(exp_theta1,2)*(pow(exp_theta2, 4) * c0 + 2*pow(exp_theta2,2) * g1 + g2);

	#ifdef PRINT_MSG
		/*std::cout << "pow(exp_theta1,2) : \n" << pow(exp_theta1,2) << std::endl;
		std::cout << "pow(exp_theta2,2) : \n" << pow(exp_theta2,2) << std::endl;
		std::cout << "c0 : \n" << c0.block(0,0,10,10) << std::endl;
        std::cout << "g1 : \n" << g1.block(0,0,10,10) << std::endl;
        std::cout << "g2 : \n" << g2.block(0,0,10,10) << std::endl;*/
    #endif

	// extract triplet indices and insert into Qx
} 


void construct_Q_spat_temp(SpMat& Qst, Vect theta, SpMat& c0, SpMat& g1, SpMat& g2, SpMat& g3,\
									  SpMat& M0, SpMat& M1, SpMat& M2){

	std::cout << "theta : " << theta.transpose() << std::endl;

	double exp_theta1 = exp(theta[0]);
	double exp_theta2 = exp(theta[1]);
	double exp_theta3 = exp(theta[2]);

	/*double exp_theta1 = exp(-5.594859);
	double exp_theta2 = exp(1.039721);
	double exp_theta3 = exp(3.688879);*/

	//std::cout << "exp(theta) : " << exp(theta[0]) << " " << exp_theta1 << " " << exp_theta2 << " " << exp_theta3 << " " << std::endl;	

	// g^2 * fem$c0 + fem$g1
	SpMat q1s = pow(exp_theta2, 2) * c0 + g1;

	 // g^4 * fem$c0 + 2 * g^2 * fem$g1 + fem$g2
		SpMat q2s = pow(exp_theta2, 4) * c0 + 2 * pow(exp_theta2,2) * g1 + g2;

		// g^6 * fem$c0 + 3 * g^4 * fem$g1 + 3 * g^2 * fem$g2 + fem$g3
		SpMat q3s = pow(exp_theta2, 6) * c0 + 3 * pow(exp_theta2,4) * g1 + 3 * pow(exp_theta2,2) * g2 + g3;

		#ifdef PRINT_MSG
			/*std::cout << "theta u : " << exp_theta1 << " " << exp_theta2 << " " << exp_theta3 << std::endl;
		std::cout << "pow(exp_theta1,2) : \n" << pow(exp_theta1,2) << std::endl;
		std::cout << "pow(exp_theta2,2) : \n" << pow(exp_theta2,2) << std::endl;
		std::cout << "q1s : \n" << q1s.block(0,0,10,10) << std::endl;
        std::cout << "q2s : \n" << q2s.block(0,0,10,10) << std::endl;
        std::cout << "q3s : \n" << q3s.block(0,0,10,10) << std::endl;*/
		#endif

		// assemble overall precision matrix Q.st
		Qst = pow(exp_theta1,2)*(KroneckerProductSparse<SpMat, SpMat>(M0, q3s) + exp_theta3 *KroneckerProductSparse<SpMat, SpMat>(M1, q2s) + pow(exp_theta3, 2)* KroneckerProductSparse<SpMat, SpMat>(M2, q1s));

		//std::cout << "Qst : \n" << Qst.block(0,0,10,10) << std::endl;
}

void construct_Qprior(SpMat& Qx, int ns, int nt, int nss, int nb, Vect theta, SpMat& c0, SpMat& g1, SpMat& g2, SpMat& g3,\
									  SpMat& M0, SpMat& M1, SpMat& M2)
{
	size_t n = ns*nt + nss + nb;

	int nst = ns*nt;
    int nu  = nst + nss;

	SpMat Q_b = 1e-5*Eigen::MatrixXd::Identity(nb, nb).sparseView(); 
	/*std::cout << "Q_b " << std::endl;
	std::cout << Eigen::MatrixXd(Q_b) << std::endl;*/

	if(ns > 0){
		SpMat Qst(nst, nst);
        SpMat Qss(nss, nss);
		// TODO: find good way to assemble Qx

		if(nt > 1){
			construct_Q_spat_temp(Qst, theta(seq(1,3)), c0, g1, g2, g3, M0, M1, M2);
            if(nss > 0){
                construct_Q_spatial(Qss, theta(seq(4,5)), c0, g1, g2);
            }
		} else {	
			construct_Q_spatial(Qst, theta(seq(1,2)), c0, g1, g2);
		}	

        //std::cout << "Qst : \n" << Qst.block(0,0,10,10) << std::endl;
        if(nss > 0){
            std::cout << "Qss : \n" << Qss.block(0,0,10,10) << std::endl;
        }

		int nnz_st = Qst.nonZeros();
		Qx.reserve(nnz_st);

		for (int k=0; k<Qst.outerSize(); ++k)
		  for (SparseMatrix<double>::InnerIterator it(Qst,k); it; ++it)
		  {
		    Qx.coeffRef(it.row(),it.col()) = it.value();                 
		  }

        if(nss > 0){
            for (int k=0; k<Qss.outerSize(); ++k)
		        for (SparseMatrix<double>::InnerIterator it(Qss,k); it; ++it)
		        {
		            Qx.coeffRef(it.row()+nst,it.col()+nst) = it.value();                 
		        }

        }
    }

    for(int i=nu; i<(n); i++){
        Qx.coeffRef(i,i) = 1e-5;
    }

    Qx.makeCompressed();
    
}


void construct_Q(SpMat& Q, int ns, int nt, int nss, int nb, Vect theta, SpMat& c0, SpMat& g1, SpMat& g2, SpMat& g3,\
									  SpMat& M0, SpMat& M1, SpMat& M2, SpMat& Ax){
   
    int n = ns*nt + nss + nb;
	double exp_theta0 = exp(theta[0]);
    SpMat Qx(n,n);

    construct_Qprior(Qx, ns, nt, nss, nb, theta, c0, g1, g2, g3, M0, M1, M2);

#ifdef PRINT_MSG
        //std::cout << "Qx : \n" << Qx.block(0,0,10,10) << std::endl;
        //std::cout << "Ax : \n" << Ax.block(0,0,10,10) << std::endl;
#endif

    Q =  Qx + exp_theta0 * Ax.transpose() * Ax;

#ifdef PRINT_MSG
        std::cout << "exp(theta0) : " << exp_theta0 << std::endl;
        std::cout << "Qx dim : " << Qx.rows() << " " << Qx.cols() << std::endl;

        std::cout << "Q  dim : " << Q.rows() << " "  << Q.cols() << std::endl;
        //std::cout << "Q : \n" << Q.block(0,0,10,10) << std::endl;
        std::cout << "theta : \n" << theta.transpose() << std::endl;

#endif
	
	/*std::cout << "Q -  exp(theta)*B'*B " << std::endl;
	std::cout << Eigen::MatrixXd(*Q) - exp_theta*B.transpose()*B << std::endl;*/

}


void eigenSolver(SpMat& Q, Vect& rhs, Vect& sol, double& logDet){

    int n = Q.rows();

    //SimplicialLLT<SpMat, Eigen::Lower, Eigen::NaturalOrdering<int>> solverQ;
    SimplicialLLT<SpMat> solverQ;
    solverQ.compute(Q);

   if(solverQ.info()!=Success) {
     cout << "Oh: Very bad" << endl;
   }

   SpMat L = solverQ.matrixL();
   if(n < 20){
        std:cout << "L: \n" << MatrixXd(L) << std::endl;
    }

   // compute log sum by hand
   logDet = 0.0;
   for(int i = 0; i<n; i++){
        logDet += log(L.coeff(i,i));
   }
   logDet *=2.0;

   //std::cout << "diag(L Eigen) : " << L.diagonal().transpose() << std::endl;
   std::cout << "log Det : " << logDet << std::endl;

   sol = solverQ.solve(rhs);

}


int pardisoSolver_factorize(SpMat& Q, double& logDet){

    SpMat Q_lower = Q.triangularView<Lower>(); 
    int nnz = Q_lower.nonZeros();
    int n = Q_lower.rows();

#ifdef PRINT_MSG
    std::cout << "in PardisoSolver. nnz = " << nnz << ", n = " << n << std::endl;
    //std::cout << "test mat lower : \n" << Q_lower << std::endl;
#endif

    int* ia; 
    int* ja;
    double* a; 

    // allocate memory
    ia = new int [n+1];
    ja = new int [nnz];
    a = new double [nnz];

    double* b;

    Q_lower.makeCompressed();

    for (int i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
         //std::cout << "ia[" << i << "] = " << ia[i] << std::endl;
    } 

    for (int i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
            //std::cout << "ja[" << i << "] = " << ja[i] << std::endl;
    }  

    for (int i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
            //std::cout << "a[" << i << "] = " << a[i] << std::endl;  
    }

#ifdef PRINT_MSG
    std::cout << "Setting up PARDISO parameters." << std::endl;
#endif

    // =========================================================================== //
    int nrhs = 1;          /* Number of right hand sides. */

  // must choose -2 for iterative solver
  int      mtype = -2;        /* Symmetric positive definite matrix */

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
  int      k;

  double   ddum;              /* Double dummy */
  int      idum;              /* Integer dummy. */
   
/* -------------------------------------------------------------------- */
/* ..  Setup Pardiso control parameters.                                */
/* -------------------------------------------------------------------- */

#ifdef PRINT_MSG
    std::cout << "Calling PARDISO init." << std::endl;
#endif

    error  = 0;
    solver = 0;

    pardisoinit (pt,  &mtype, &solver, iparm, dparm, &error);

    //std::cout << "error : " << error << std::endl;

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

#ifdef PRINT_MSG
    std::cout << "[PARDISO]: License check was successful ... " << std::endl;
#endif

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

    for (int i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (int i = 0; i < nnz; i++) {
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

#ifdef PRINT_MSG
    std::cout << "After PARDISO checks." << std::endl;
#endif

    iparm[19-1] = -1; // in order to compute Gflops
    printf("\nGFlops factorisation : %i", iparm[19-1]);

    // start timer phase 1
    double timespent_p11 = -omp_get_wtime();

    phase = 11; 

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
         &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error, dparm);
    
    if (error != 0) {
        printf("\nERROR during symbolic factorization: %d", error);
        exit(1);
    }

    // get time phase 1
    timespent_p11 += omp_get_wtime();
    // printf("\nTime spent on Phase 1 : %f", time_spent_p11);
   
/* -------------------------------------------------------------------- */
/* ..  Numerical factorization.                                         */
/* -------------------------------------------------------------------- */

    // start timer phase 2
    double timespent_p22 = -omp_get_wtime();

    phase = 22;
    iparm[32] = 1; /* compute determinant */

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }

    // get time phase 2
    timespent_p22 += omp_get_wtime();

    // printf("\nFactorization completed ...\n ");
    printf("\nFactorization completed .. \n");

    logDet = dparm[32];
    printf("\nPardiso   log(det) = %f ", logDet);

    int gflops_fact = iparm[19-1];
    int mem_fact_solve = iparm[17-1];

    printf("\nGFlops factorisation : %i", iparm[19-1]);
    printf("\nMem fact             : %i\n", mem_fact_solve);

    return 1;

}

int pardisoSolver_factorizeSolve(SpMat& Q, Vect& rhs, Vect& sol, double& logDet){

    SpMat Q_lower = Q.triangularView<Lower>(); 
    int nnz = Q_lower.nonZeros();
    int n = Q_lower.rows();

#ifdef PRINT_MSG
    std::cout << "in PardisoSolver. nnz = " << nnz << ", n = " << n << std::endl;
    //std::cout << "test mat lower : \n" << Q_lower << std::endl;
#endif

    int* ia; 
    int* ja;
    double* a; 

    // allocate memory
    ia = new int [n+1];
    ja = new int [nnz];
    a = new double [nnz];

    double* b = new double [n];
    // empty solution vector
    double* x = new double [n];

    for (int i = 0; i < n; ++i){
        b[i] = rhs[i];
    }

    Q_lower.makeCompressed();

    for (int i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
         //std::cout << "ia[" << i << "] = " << ia[i] << std::endl;
    } 

    for (int i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
            //std::cout << "ja[" << i << "] = " << ja[i] << std::endl;
    }  

    for (int i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
            //std::cout << "a[" << i << "] = " << a[i] << std::endl;  
    }

#ifdef PRINT_MSG
    std::cout << "Setting up PARDISO parameters." << std::endl;
#endif

    // =========================================================================== //
    int nrhs = 1;          /* Number of right hand sides. */

  // must choose -2 for iterative solver
  int      mtype = -2;        /* Symmetric positive definite matrix */

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
  int      k;

  double   ddum;              /* Double dummy */
  int      idum;              /* Integer dummy. */
   
/* -------------------------------------------------------------------- */
/* ..  Setup Pardiso control parameters.                                */
/* -------------------------------------------------------------------- */

#ifdef PRINT_MSG
    std::cout << "Calling PARDISO init." << std::endl;
#endif

    error  = 0;
    solver = 0;

    pardisoinit (pt,  &mtype, &solver, iparm, dparm, &error);

    //std::cout << "error : " << error << std::endl;

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

#ifdef PRINT_MSG
    std::cout << "[PARDISO]: License check was successful ... " << std::endl;
#endif

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

    for (int i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (int i = 0; i < nnz; i++) {
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

#ifdef PRINT_MSG
    std::cout << "After PARDISO checks." << std::endl;
#endif

    iparm[19-1] = -1; // in order to compute Gflops
    printf("\nGFlops factorisation : %i", iparm[19-1]);

    // start timer phase 1
    double timespent_p11 = -omp_get_wtime();

    phase = 11; 

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
         &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error, dparm);
    
    if (error != 0) {
        printf("\nERROR during symbolic factorization: %d", error);
        exit(1);
    }

    // get time phase 1
    timespent_p11 += omp_get_wtime();
    // printf("\nTime spent on Phase 1 : %f", time_spent_p11);
   
/* -------------------------------------------------------------------- */
/* ..  Numerical factorization.                                         */
/* -------------------------------------------------------------------- */

    // start timer phase 2
    double timespent_p22 = -omp_get_wtime();

    phase = 22;
    iparm[32] = 1; /* compute determinant */

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }

    // get time phase 2
    timespent_p22 += omp_get_wtime();

    // printf("\nFactorization completed ...\n ");
    printf("\nFactorization completed .. \n");

    logDet = dparm[32];
    printf("\nPardiso   log(det) = %f ", logDet);

    int gflops_fact = iparm[19-1];
    int mem_fact_solve = iparm[17-1];

    printf("\nGFlops factorisation : %i", iparm[19-1]);
    printf("\nMem fact + solve     : %i\n", mem_fact_solve);

/* -------------------------------------------------------------------- */    
/* ..  Back substitution and iterative refinement.                      */
/* -------------------------------------------------------------------- */    

    // start timer phase 3
    //double timespent_p33 = 0;
    double timespent_p33 = -omp_get_wtime();

    phase = 33;

    iparm[7] = 0;       /* Max numbers of iterative refinement steps. */
 
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, b, x, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during solution: %d", error);
        exit(3);
    }

    // get time phase 3
    timespent_p33 += omp_get_wtime(); 

#ifdef PRINT_MSG
    printf("\nSolve completed ... "); 
#endif

    for(int i = 0; i<n; i++){
        sol[i] = x[i];
    }

    return 1;

}

// function to evaluate MVN
// evaluate in x, mu is the mean
// log((2*pi)^(n/2)) + log(|Q|^(1/2)) - 0.5*(x - mu)^T*Q*(x - mu)
double evaluate_MVN(SpMat& Q, double& logDet, Vect& mu, Vect& x){
    int n = Q.rows();

    // - 0.5*(x - mu)^T*Q*(x - mu)
    double quad_term = 0.5 * (mu - x).transpose() * Q * (mu - x);  
    printf("quad term: %f\n", quad_term); 
    double value     = 0.5*n*log(2*M_PI) + 0.5*logDet - quad_term;

    return value; 
}

// log((2*pi)^(n/2)) + log(|Q|^(1/2)) - 0.5*(x - y)^T*Q*(x - y) which becomes
// log((2*pi)^(n/2)) + 0.5 * no * log_theta_noise - 0.5*(x - y)^T*exp(log_theta_noise)*I*(x - y)
double evaluate_logLikelihood_normal(double log_theta_noise, Vect& y, Vect& Ax){

    int no = y.size();
    double logDet    = no * log_theta_noise;
    double quad_term = 0.5 * exp(log_theta_noise) * (Ax - y).transpose() * (Ax - y);  
	double val       = 0.5*no*log(2*M_PI) + 0.5*logDet - quad_term;

    return val;
}



int main(int argc, char* argv[])
{

size_t i; // iteration variable

#if 0

    /*
    int ns=1;
    int nt=6;
    int nb=1;
    int no=0;

    int n = ns*nt+nb;

    //SpMat Q = gen_test_mat_base1();
    SpMat Q = gen_test_mat_base2();
    std::cout << "Q: \n" << Q << std::endl;
    */

    int ns=3;
    int nt=3;
    int nss=1;
    int nb=3;
    //int n = ns*nt + nss + nb;

    SpMat Q = gen_test_mat_base4(ns, nt, nss, nb);
    //SpMat Q = gen_test_mat_base4_prior(ns, nt, nss);

    //std::cout << "Q : \n" << MatrixXd(Q) << std::endl;


    //Vect rhs(n);
    //rhs.setOnes(n);

    //exit(1);



#else

    if(argc != 1 + 7){
        std::cout << "wrong number of input parameters. " << std::endl;

        std::cerr << "INLA Call : ns nt nb no path/to/files solver_type" << std::endl;

        std::cerr << "[integer:ns]                number of spatial grid points " << std::endl;
        std::cerr << "[integer:nt]                number of temporal grid points " << std::endl;
        std::cerr << "[integer:nss]               number of spatial grid points add. spatial field " << std::endl;
        std::cerr << "[integer:nb]                number of fixed effects" << std::endl;
        std::cerr << "[integer:no]                number of data samples" << std::endl;

        std::cerr << "[string:base_path]          path to folder containing matrix files " << std::endl;

        std::cerr << "[string:solver_type]        BTA or PARDISO" << std::endl;
    

        exit(1);
    }

    std::cout << "reading in example. " << std::endl;

    size_t ns  = atoi(argv[1]);
    size_t nt  = atoi(argv[2]);
    size_t nss = atoi(argv[3]);
    size_t nb  = atoi(argv[4]);
    std::cout << "ns = " << ns << ", nt = " << nt << ", nb = " << nb << std::endl;
    size_t no = atoi(argv[5]);
    //std::string no_s = argv[5];
    // to be filled later

    // set nt = 1 if ns > 0 & nt = 0
    if(ns > 0 && nt == 0){
        nt = 1;
    } 

    // also save as string
    std::string ns_s = std::to_string(ns);
    std::string nt_s = std::to_string(nt);
    std::string nb_s = std::to_string(nb);
    std::string no_s = std::to_string(no); 
    std::string n_s  = std::to_string(ns*nt + nss + nb);

    std::string base_path = argv[6];    

    std::string solver_type = argv[7];
    // check if solver type is neither PARDISO nor RGF :
    if(solver_type.compare("PARDISO") != 0 && solver_type.compare("BTA") != 0){
        std::cout << "Unknown solver type. Available options are :\nPARDISO\nBTA" << std::endl;
        exit(1);
    }

    /* ---------------- read in matrices ---------------- */

    // dimension hyperparamter vector
    int dim_th;

    // spatial component
    SpMat c0; 
    SpMat g1; 
    SpMat g2;

    // spatial-temporal parts
    SpMat g3;
    SpMat M0;
    SpMat M1;
    SpMat M2;

    // data component / fixed effects
    MatrixXd B;
    SpMat Ax; 
    Vect y;

    if(ns == 0 && nt == 0){

        dim_th = 1;

        // read in design matrix 
        // files containing B
        std::string B_file        =  base_path + "/B_" + no_s + "_" + nb_s + ".dat";
        file_exists(B_file); 

        // casting no_s as integer
        no = std::stoi(no_s);
        std::cout << "total number of observations : " << no << std::endl;
      
        B = read_matrix(B_file, no, nb);

        // std::cout << "y : \n"  << y << std::endl;    
        // std::cout << "B : \n" << B << std::endl;

    } else if(ns > 0 && nt == 1){

        std::cout << "spatial model." << std::endl;

        dim_th = 3;

        // check spatial FEM matrices
        std::string c0_file       =  base_path + "/c0_" + ns_s + ".dat";
        file_exists(c0_file);
        std::string g1_file       =  base_path + "/g1_" + ns_s + ".dat";
        file_exists(g1_file);
        std::string g2_file       =  base_path + "/g2_" + ns_s + ".dat";
        file_exists(g2_file);

        // check projection matrix for A.st
        std::string Ax_file     =  base_path + "/Ax_" + no_s + "_" + n_s + ".dat";
        file_exists(Ax_file);

        // read in matrices
        c0 = read_sym_CSC(c0_file);
        g1 = read_sym_CSC(g1_file);
        g2 = read_sym_CSC(g2_file);

        // doesnt require no to be read, can read no from Ax
        Ax = readCSC(Ax_file);
        // get rows from the matrix directly
        // doesnt work for B
        no = Ax.rows();
        std::cout << "total number of observations : " << no << std::endl;


        /*std::cout << "g1 : \n" << g1.block(0,0,10,10) << std::endl;
        std::cout << "g2 : \n" << g2.block(0,0,10,10) << std::endl;
        std::cout << "Ax : \n" << Ax.block(0,0,10,10) << std::endl;*/

    } else if(ns > 0 && nt > 1) {

        std::cout << "spatial-temporal model. Reading in matrices." << std::endl;

        if(nss == 0){
            dim_th = 4;
        } else if(nss > 0){
            dim_th = 6;
        } else {
            printf("nss invalid!! nss = %ld\n", nss);
            exit(1);
        }

        // files to construct Q.u depending on HYPERPARAMETERS theta
        std::string c0_file      =  base_path + "/c0_" + ns_s + ".dat";
        file_exists(c0_file);
        std::string g1_file      =  base_path + "/g1_" + ns_s + ".dat";
        file_exists(g1_file);
        std::string g2_file      =  base_path + "/g2_" + ns_s + ".dat";
        file_exists(g2_file);
        std::string g3_file      =  base_path + "/g3_" + ns_s + ".dat";
        file_exists(g3_file);

        std::string M0_file      =  base_path + "/M0_" + nt_s + ".dat";
        file_exists(M0_file);
        std::string M1_file      =  base_path + "/M1_" + nt_s + ".dat";
        file_exists(M1_file);
        std::string M2_file      =  base_path + "/M2_" + nt_s + ".dat";
        file_exists(M2_file);  

        // check projection matrix for A.st
        std::string Ax_file     =  base_path + "/Ax_" + no_s + "_" + n_s + ".dat";
        file_exists(Ax_file);

        // read in matrices
        c0 = read_sym_CSC(c0_file);
        g1 = read_sym_CSC(g1_file);
        g2 = read_sym_CSC(g2_file);
        g3 = read_sym_CSC(g3_file);

        M0 = read_sym_CSC(M0_file);
        //arma::mat(M0).submat(0,0,nt-1,nt-1).print();
        M1 = read_sym_CSC(M1_file);
        //arma::mat(M1).submat(0,0,nt-1,nt-1).print();
        M2 = read_sym_CSC(M2_file);
        //arma::mat(M2).submat(0,0,nt-1,nt-1).print();

        Ax = readCSC(Ax_file);
        // get rows from the matrix directly
        // doesnt work for B
        no = Ax.rows();
        std::cout << "total number of observations : " << no << std::endl;

    } else {
        std::cout << "invalid parameters : ns nt !!" << std::endl;
        exit(1);
    }

    // data y
    std::string y_file        =  base_path + "/y_" + no_s + "_1" + ".dat";
    file_exists(y_file);
    // at this point no is set ... 
    // not a pretty solution. 
    y = read_matrix(y_file, no, 1);

    /* ----------------------- initialise random theta -------------------------------- */

    Vect theta_RINLA(dim_th);
    Vect theta_INLAdist(dim_th);
    Vect theta_prior(dim_th);

	if(nt == 1){
	    //theta << -1.5,-5,-2;
	    //theta.print();
        printf("only meant for nt > 0 at the moment.\n");
        exit(1);
  	} else {
        // set list with different theta values -> from RINLA & INLA_DIST
        // show overview table with values somewhere, iterate through those, read in files
        std::string theta_RINLA_file = base_path + "/theta_RINLA_" + to_string(dim_th) + "_1" + ".txt";
        theta_RINLA = read_matrix(theta_RINLA_file, dim_th, 1);
        std::cout << "theta_RINLA    : " << theta_RINLA.transpose() << std::endl;

        std::string theta_INLAdist_file = base_path + "/theta_INLAdist_" + to_string(dim_th) + "_1" + ".txt";
        theta_INLAdist = read_matrix(theta_INLAdist_file, dim_th, 1);
        std::cout << "theta_INLADIST : " << theta_INLAdist.transpose() << std::endl;
    
  	}

    printf("# threads: %d\n", omp_get_max_threads());

#endif

    size_t n = ns*nt + nss + nb;
    //size_t n = ns*nt + nss;
    SpMat Qx(n,n);
    SpMat Q(n,n);
    Vect rhs(n);
    Vect mu(n);
    Vect x(n);
    Vect zeroVec = Vect::Zero(n);
    double logDet;

    // ****************************************************************************** //  
    // R-INLA : marginal likelihood compute components using theta_RINLA
    // log p(x | \theta) + log p(y | x, \theta) - log p(x | \theta, y )
    // => compute x at the mode and plug it in. 
    std::cout << "\nComputing marginal log-likelihood for theta_RINLA." << std::endl;  

    // ****** log p(x | \theta, y ) ******* //
    double exp_theta_RINLA = exp(theta_RINLA[0]);
    rhs = exp_theta_RINLA*Ax.transpose()*y;
    
    construct_Q(Q, ns, nt, nss, nb, theta_RINLA, c0, g1, g2, g3, M0, M1, M2, Ax);
    //std::cout << "Q(theta_RINLA) : \n" << Q.block(0,0,10,10) << std::endl;

    //eigenSolver(Q, rhs, sol, logDet);
    pardisoSolver_factorizeSolve(Q, rhs, mu, logDet);
    x = mu;
    double logPcondVal_RINLA = evaluate_MVN(Q, logDet, mu, x);
    printf("log p(x | theta_RINLA, y ) = %f\n", logPcondVal_RINLA);

    // ****** log p(x | \theta ) ******* //
    construct_Qprior(Qx, ns, nt, nss, nb, theta_RINLA, c0, g1, g2, g3, M0, M1, M2);
    //std::cout << "Qx(theta_RINLA) : \n" << Qx.block(0,0,10,10) << std::endl;

    pardisoSolver_factorize(Qx, logDet);
    double logPpriorLat_RINLA = evaluate_MVN(Q, logDet, zeroVec, mu);
    printf("log p(x | theta_RINLA) = %f\n", logPpriorLat_RINLA);

    // ******  log p(y | x, \theta) ******* //
    Vect Axx_RINLA = Ax*x;
    double logPlik_RINLA = evaluate_logLikelihood_normal(theta_RINLA[0], y, Axx_RINLA);

    // ******  sum  ******* //
    double mll_RINLA = logPpriorLat_RINLA + logPlik_RINLA - logPcondVal_RINLA;
    printf("marg. log likelihood RINLA: %f\n", mll_RINLA);

    // ****************************************************************************** //
    // INLA_DIST : marginal likelihood compute components using theta_INLAdist
    std::cout << "\nComputing marginal log-likelihood for theta_INLAdist." << std::endl;  
    double exp_theta_INLAdist = exp(theta_INLAdist[0]);

    construct_Q(Q, ns, nt, nss, nb, theta_INLAdist, c0, g1, g2, g3, M0, M1, M2, Ax);
    //std::cout << "Q(theta_INLAdist) : \n" << Q.block(0,0,10,10) << std::endl;

    //eigenSolver(Q, rhs, sol, logDet);
    pardisoSolver_factorizeSolve(Q, rhs, mu, logDet);
    x = mu;
    double logPcondVal_INLAdist = evaluate_MVN(Q, logDet, mu, x);
    printf("log p(x | theta_INLAdist, y ) = %f\n", logPcondVal_INLAdist);

    construct_Qprior(Qx, ns, nt, nss, nb, theta_INLAdist, c0, g1, g2, g3, M0, M1, M2);
    //std::cout << "Qx(theta_INLAdist) : \n" << Qx.block(0,0,10,10) << std::endl;

    // ****** log p(x | \theta ) ******* //
    construct_Qprior(Qx, ns, nt, nss, nb, theta_INLAdist, c0, g1, g2, g3, M0, M1, M2);
    //std::cout << "Qx(theta_RINLA) : \n" << Qx.block(0,0,10,10) << std::endl;

    pardisoSolver_factorize(Qx, logDet);
    double logPpriorLat_INLAdist = evaluate_MVN(Q, logDet, zeroVec, mu);
    printf("log p(x | theta_RINLA) = %f\n", logPpriorLat_INLAdist);

    // ******  log p(y | x, \theta) ******* //
    Vect Axx_INLAdist = Ax*x;
    double logPlik_INLAdist = evaluate_logLikelihood_normal(theta_INLAdist[0], y, Axx_INLAdist);

    // ******  sum  ******* //
    double mll_INLAdist = logPpriorLat_INLAdist + logPlik_INLAdist - logPcondVal_INLAdist;
    printf("marg. log likelihood INLAdist: %f\n", mll_INLAdist);


#if 0
    // true inv diag from Eigen
    //SimplicialLLT<SpMat, Eigen::Lower, Eigen::NaturalOrdering<int>> solverQ;
    SimplicialLLT<SpMat> solverQ;
    solverQ.compute(Q);

   if(solverQ.info()!=Success) {
     cout << "Oh: Very bad" << endl;
   }

   SpMat L = solverQ.matrixL();
   if(n < 20){
        std:cout << "L: \n" << MatrixXd(L) << std::endl;
    }

   SpMat eye(n,n);
   eye.setIdentity();

   // compute log sum by hand
   double logDetEigen = 0.0;
   for(int i = 0; i<n; i++){
        logDetEigen += log(L.coeff(i,i));
   }
   logDetEigen *=2.0;

   //std::cout << "diag(L Eigen) : " << L.diagonal().transpose() << std::endl;
   std::cout << "log Det Eigen : " << logDetEigen << std::endl;

   //SpMat inv_Q = solverQ.solve(eye);
   //MatrixXd inv_Q_dense = MatrixXd(inv_Q.triangularView<Lower>());
   //std::cout << "inv(Q)\n" << inv_Q_dense << std::endl;

#endif  // end #if 1 simplicial solver
      
  return 0;


}
