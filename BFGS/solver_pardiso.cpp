
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <fstream>
#include <iostream>

#include <omp.h>
#include <time.h>

#include <armadillo>

// C++ compatible
using namespace std;

//#define PRINT_PAR

using namespace Eigen;

// typedef Eigen::Matrix<double, Dynamic, Dynamic> Mat;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::CholmodSimplicialLDLT  <SpMat > Solver;
typedef Eigen::VectorXd Vector;




/* PARDISO prototype. */
extern "C" void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
extern "C" void pardiso     (void   *, int    *,   int *, int *,    int *, int *, 
                  double *, int    *,    int *, int *,   int *, int *,
                     int *, double *, double *, int *, double *);
extern "C" void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
extern "C" void pardiso_chkvec     (int *, int *, double *, int *);
extern "C" void pardiso_printstats (int *, int *, double *, int *, int *, int *,
                           double *, int *);

/* ---------------------------------------------------------------------------------------------- */

void log_det_cholmod(SpMat *A, double *log_det)
{

	Solver solver;
	solver.analyzePattern(*A);
	solver.factorize(*A);

	*log_det = solver.logDeterminant();

	//std::cout << "solution vector u : " << *u << std::endl;

}

void solve_cholmod(SpMat *A, Vector *f, Vector& u, double *log_det)
{

	Solver solver;
	solver.analyzePattern(*A);
	solver.factorize(*A);

	*log_det = solver.logDeterminant();

	u = solver.solve(*f);

	//std::cout << "solution vector u : " << *u << std::endl;

}

void extract_inv_diag(SpMat& Q, Vector& vars){

		MatrixXd Q_dense = MatrixXd(Q);
		//std::cout << "Q dense :\n" << Q_dense << std::endl; 

		MatrixXd Q_inv = Q_dense.inverse();		
		//std::cout << "Q inv :\n" << Q_inv << std::endl; 

		vars = Q_inv.diagonal();


}

void compute_inverse(MatrixXd& Q, MatrixXd& Q_inv){

		Q_inv = Q.inverse();		
		//std::cout << "Q inv :\n" << Q_inv << std::endl; 


}

/* ---------------------------------------------------------------------------------------------- */

int log_det_pardiso(SpMat *A, double *log_det_A){

	// get everything into the right format

	// only take lower triangular part of A
	SpMat A_lower = A->triangularView<Lower>(); 

	// this time require CSR format

	unsigned int nnz = A_lower.nonZeros();
	//std::cout << "number of non zeros : " << nnz << std::endl;

	int* ia; 
	int* ja;
	double* a; 

	// allocate memory
	ia = new int [n+1];
	ja = new int [nnz];
	a = new double [nnz];

	A_lower.makeCompressed();

	for (int i = 0; i < n+1; ++i){
		ia[i] = A_lower.outerIndexPtr()[i]; 
	}  

	for (int i = 0; i < nnz; ++i){
		ja[i] = A_lower.innerIndexPtr()[i];
	}  

	for (int i = 0; i < nnz; ++i){
		a[i] = A_lower.valuePtr()[i];
	}  

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
	int      i, k;

	double   ddum;              /* Double dummy */
	int      idum;              /* Integer dummy. */


	/* -------------------------------------------------------------------- */
	/* ..  Setup Pardiso control parameters.                                */
	/* -------------------------------------------------------------------- */

	error = 0;
	solver=0;

	if(solver == 1){
	  mtype = -2;
	  std::cout << "Changed matrix type to -2 for iterative solver !\n" << std::endl;
	}

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

	maxfct = 1;     /* Maximum number of numerical factorizations.  */
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
	/* ..  Reordering and Symbolic Factorization.  This step also allocates */
	/*     all memory that is necessary for the factorization.              */
	/* -------------------------------------------------------------------- */

	iparm[19-1] = -1; // in order to compute Gflops
	#ifdef PRINT_PAR
		printf("\nGFlops factorisation : %i", iparm[19-1]);
	#endif

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
	#ifdef PRINT_PAR
		printf("\nFactorization completed .. \n");
	#endif

	*log_det_A = dparm[32];

	#ifdef PRINT_PAR
		printf("\nPardiso   log(det) = %f ", *log_det_A);	
	#endif

	int gflops_fact = iparm[19-1];
	int mem_fact_solve = iparm[17-1];

	#ifdef PRINT_PAR
		printf("\nGFlops factorisation : %i", iparm[19-1]);
		printf("\nMem fact + solve     : %i", mem_fact_solve);
	#endif

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
	/* ..  Print statistics                                                 */
	/* -------------------------------------------------------------------- */   

	#ifdef PRINT_PAR
		printf("\nTime spent on phase 1 : %f s", timespent_p11);
		printf("\nTime spent on phase 2 : %f s", timespent_p22);
	#endif

	/* -------------------------------------------------------------------- */    
	/* ..  Termination and release of memory.                               */
	/* -------------------------------------------------------------------- */    
	phase = -1;                 /* Release internal memory. */

	pardiso (pt, &maxfct, &mnum, &mtype, &phase,
	         &n, &ddum, ia, ja, &idum, &nrhs,
	         iparm, &msglvl, &ddum, &ddum, &error,  dparm);


	delete[] ia;
	delete[] ja;
	delete[] a;

	return 0;         

} 


int solve_pardiso(SpMat *A, Vector *f, Vector& u, double *log_det_A){

	std::string base_path = "/home/x_gaedkelb/b_INLA/BFGS";

	// get everything into the right format

	// only take lower triangular part of A
	SpMat A_lower = A->triangularView<Lower>(); 

	// this time require CSR format

	int n = f->size();
	//std::cout << "dim n : " << n << std::endl;

	unsigned int nnz = A_lower.nonZeros();
	//std::cout << "number of non zeros : " << nnz << std::endl;

	int* ia; 
	int* ja;
	double* a; 

	// allocate memory
	ia = new int [n+1];
	ja = new int [nnz];
	a = new double [nnz];

	A_lower.makeCompressed();

	for (int i = 0; i < n+1; ++i){
		ia[i] = A_lower.outerIndexPtr()[i]; 
	}  

	for (int i = 0; i < nnz; ++i){
		ja[i] = A_lower.innerIndexPtr()[i];
	}  

	for (int i = 0; i < nnz; ++i){
		a[i] = A_lower.valuePtr()[i];
	}  

	double* b = new double [n];

	for (int i = 0; i < n; ++i){
		b[i] = (*f)(i);
	}  

	// empty solution vector
	double* x = new double [n];

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
	int      i, k;

	double   ddum;              /* Double dummy */
	int      idum;              /* Integer dummy. */


	/* -------------------------------------------------------------------- */
	/* ..  Setup Pardiso control parameters.                                */
	/* -------------------------------------------------------------------- */

	error = 0;
	solver=0;

	if(solver == 1){
	  mtype = -2;
	  std::cout << "Changed matrix type to -2 for iterative solver !\n" << std::endl;
	}

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

	maxfct = 1;     /* Maximum number of numerical factorizations.  */
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

	iparm[19-1] = -1; // in order to compute Gflops
	#ifdef PRINT_PAR
		printf("\nGFlops factorisation : %i", iparm[19-1]);
	#endif

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
	#ifdef PRINT_PAR
		printf("\nFactorization completed .. \n");
	#endif

	*log_det_A = dparm[32];
	#ifdef PRINT_PAR
		printf("\nPardiso   log(det) = %f ", *log_det_A);	
	#endif

	int gflops_fact = iparm[19-1];
	int mem_fact_solve = iparm[17-1];

	#ifdef PRINT_PAR
		printf("\nGFlops factorisation : %i", iparm[19-1]);
		printf("\nMem fact + solve     : %i", mem_fact_solve);
	#endif

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

	#ifdef PRINT_PAR
		printf("\nSolve completed ... "); 
	#endif

	for (i = 0; i < n; i++) {
	   u[i] = x[i];
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
	/* ..  Print statistics                                                 */
	/* -------------------------------------------------------------------- */   

	#ifdef PRINT_PAR
		printf("\nTime spent on phase 1 : %f s", timespent_p11);
		printf("\nTime spent on phase 2 : %f s", timespent_p22);
		printf("\nTime spent on phase 3 : %f s\n", timespent_p33);
	#endif

	/* -------------------------------------------------------------------- */    
	/* ..  Termination and release of memory.                               */
	/* -------------------------------------------------------------------- */    
	phase = -1;                 /* Release internal memory. */

	pardiso (pt, &maxfct, &mnum, &mtype, &phase,
	         &n, &ddum, ia, ja, &idum, &nrhs,
	         iparm, &msglvl, &ddum, &ddum, &error,  dparm);


	delete[] ia;
	delete[] ja;
	delete[] a;
	delete[] x;
	delete[] b;

	return 0;         

} 


