
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

// COMPUTES LOG DETERMINANT OF A USING PARDISO
void log_det_pardiso(SpMat *A, double &log_det_A){

	// get everything into the right format
	int nrhs = 0;          /* Number of right hand sides. */


	// only take lower triangular part of A
	SpMat A_lower = A->triangularView<Lower>(); 

	// this time require CSR format

	int n = A->rows();

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
	     exit(1);
	}
	else {
		#ifdef PRINT_PAR
	    	printf("[PARDISO]: License check was successful ... \n");
	    #endif
	}

	/* Numbers of processors, value of OMP_NUM_THREADS */
	// replace this to get second thread number
	var = getenv("OMP_NUM_THREADS");
	if(var != NULL)
	    sscanf( var, "%d", &num_procs );
	else {
	    printf("Set environment OMP_NUM_THREADS to 1");
	    exit(1);
	}

	// MANUALLY SET THREADS for now
	// nested omp 
	//iparm[2]  = num_procs;
	iparm[2] = 16;

	maxfct = 1;     /* Maximum number of numerical factorizations.  */
	mnum   = 1;         /* Which factorization to use. */

	// enable again to see parallelism
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

	log_det_A = dparm[32];

	#ifdef PRINT_PAR
		printf("\nPardiso   log(det) = %f ", log_det_A);	
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

} 

// COMPUTES LOG DETERMINANT OF A & SOLVES Au = f for u USING PARDISO
void solve_pardiso(SpMat *A, Vector *f, Vector& u, double &log_det_A){

	// get everything into the right format

	// only take lower triangular part of A
	SpMat A_lower = A->triangularView<Lower>(); 

	// this time require CSR format

	int n = f->size();
	#ifdef PRINT_PAR
		std::cout << "dim n : " << n << std::endl;
	#endif

	unsigned int nnz = A_lower.nonZeros();
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
	     exit(1);  
	} else {
		#ifdef PRINT_PAR
	    	printf("[PARDISO]: License check was successful ... \n");
	    #endif
	}

	/* Numbers of processors, value of OMP_NUM_THREADS */
	var = getenv("OMP_NUM_THREADS");
	if(var != NULL)
	    sscanf( var, "%d", &num_procs );
	else {
	    printf("Set environment OMP_NUM_THREADS to 1");
	    exit(1);
	}

	// MANUALLY SET THREADS
	// nested omp 
	//iparm[2]  = num_procs;
	iparm[2] = 8;

	//std::cout << "in pardiso, OMP_NUM_THREADS : " << var << std::endl;

	maxfct = 1;     /* Maximum number of numerical factorizations.  */
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

	/*pardiso_printstats (&mtype, &n, a, ia, ja, &nrhs, b, &error);
	if (error != 0) {
	    printf("\nERROR right hand side: %d", error);
	    exit(1);
	}*/

	#ifdef PRINT_PAR
		std::cout << "Passed all initial PARDISO checks." << std::endl;
	#endif

	/* -------------------------------------------------------------------- */
	/* ..  Reordering and Symbolic Factorization.  This step also allocates */
	/*     all memory that is necessary for the factorization.              */
	/* -------------------------------------------------------------------- */

	iparm[19-1] = -1; // in order to compute Gflops

	// start timer phase 1
	double timespent_p11 = -omp_get_wtime();

	phase = 11; 

	pardiso (pt, &maxfct, &mnum, &mtype, &phase,
	     &n, a, ia, ja, &idum, &nrhs,
	         iparm, &msglvl, &ddum, &ddum, &error, dparm);

	#ifdef PRINT_PAR
		printf("after symbolic factorization.\n");
	#endif

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

	#ifdef PRINT_PAR
		std::cout << "a, ia, ja : " << std::endl;
		for(int i = 0; i < 5; i ++){
			std::cout << a[i] << " " << ia[i] << " " << ja[i] << std::endl;
		}
	#endif

	pardiso (pt, &maxfct, &mnum, &mtype, &phase,
	         &n, a, ia, ja, &idum, &nrhs,
	         iparm, &msglvl, &ddum, &ddum, &error,  dparm);

	#ifdef PRINT_PAR
		printf("after factorization.\n");
	#endif

	if (error != 0) {
	    printf("\nERROR during numerical factorization: %d", error);
	    exit(2);
	}

	// get time phase 2
	timespent_p22 += omp_get_wtime();

	#ifdef PRINT_PAR
		printf("\nFactorization completed .. \n");
	#endif

	log_det_A = dparm[32];
	#ifdef PRINT_PAR
		printf("\nPardiso   log(det) = %f ", log_det_A);	
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
	   u(i) = x[i];
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

} 

// COMPUTES INVERSE DIAGONAL ELEMENTS OF A, stores them in inv_diag
void inv_diagonal_pardiso(SpMat *A, Vector& inv_diag){

	// get everything into the right format

	// only take lower triangular part of A
	SpMat A_lower = A->triangularView<Lower>(); 

	// this time require CSR format

	int n = A->rows();
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

	int nrhs = 0;          /* Number of right hand sides. */

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
	     exit(1); 
	}
	else{
		#ifdef PRINT_PAR
	    	printf("[PARDISO]: License check was successful ... \n");
	    #endif
	}

	/* Numbers of processors, value of OMP_NUM_THREADS */
	var = getenv("OMP_NUM_THREADS");
	if(var != NULL)
	    sscanf( var, "%d", &num_procs );
	else {
	    printf("Set environment OMP_NUM_THREADS to 1");
	    exit(1);
	}

	// MANUALLY SET THREADS
	// nested omp 
	//iparm[2]  = num_procs;
	iparm[2] = 16;

	maxfct = 1;     /* Maximum number of numerical factorizations.  */
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

	/*pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
	if (error != 0) {
	    printf("\nERROR in consistency of matrix: %d", error);
	    exit(1);
	}*/

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

	int gflops_fact = iparm[19-1];
	int mem_fact_solve = iparm[17-1];

	#ifdef PRINT_PAR
		printf("\nGFlops factorisation : %i", iparm[19-1]);
		printf("\nMem fact + solve     : %i", mem_fact_solve);
	#endif

	/* -------------------------------------------------------------------- */    
	/* ... Inverse factorization.                                           */                                       
	/* -------------------------------------------------------------------- */  

	double timespent_sel_inv = 0; 

	#ifdef PRINT_PAR
		printf("\nCompute Diagonal Elements of the inverse of A ... \n");
	#endif

	timespent_sel_inv = -omp_get_wtime();

	phase = -22;
	//iparm[35]  = 1; /*  no not overwrite internal factor L // crashes for larger matrices if uncommented */ 
	pardiso (pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
	      iparm, &msglvl, &ddum, &ddum, &error,  dparm);

	// get time to compute selected inverse
	timespent_sel_inv += omp_get_wtime(); 

	for (k = 0; k < n; k++){
	    int j = ia[k]-1;
	    inv_diag(k) = a[j];
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
		printf("\nTime spent on sel inv : %f s\n", timespent_sel_inv);
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

} 





