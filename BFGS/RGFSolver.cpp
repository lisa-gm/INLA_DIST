#include "RGFSolver.h"

#include "../RGF/RGF.H"


RGFSolver::RGFSolver(size_t ns, size_t nt, size_t nb, size_t no) : ns_t(ns), nt_t(nt), nb_t(nb), no_t(no){
   	
   	#ifdef PRINT_MSG
   	std::cout << "constructing RGF solver." << std::endl;
   	#endif

   	n = ns_t*nt_t + nb_t;

}

// currently not needed !!
void RGFSolver::symbolic_factorization(SpMat& Q, int& init) {
	init = 1;
	std::cout << "Placeholder SYMBOLIC_FACTORIZATION() not needed for RGF." << std::endl;
}

// NOTE: this function is written to factorize prior! Assumes tridiagonal structure.
void RGFSolver::factorize(SpMat& Q, double& log_det) {

	#ifdef PRINT_MSG
	std::cout << "in RGF FACTORIZE()." << std::endl;
	#endif

	// check if n and Q.size() match
    if((n - nb_t) != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n-nb = %d.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

    #ifdef PRINT_MSG
    	std::cout << "Q in RGFSolver.cpp : \n" << Q.block(0,0,10,10) << std::endl;
    #endif

	// only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 
    nnz = Q_lower.nonZeros();

    size_t* ia; 
    size_t* ja;
    T* a; 

    // allocate memory
    ia = new long unsigned int [n+1];
    ja = new long unsigned int [nnz];
    a = new double [nnz];

    Q_lower.makeCompressed();

    for (i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
    }  

    for (i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
    }  

    for (i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
    }

	double t_factorise;
	RGF<T> *solver;

	// SET nb_t to zero : this function is for precision matrix of prior!
	solver = new RGF<T>(ia, ja, a, ns_t, nt_t, 0);

	t_factorise = get_time(0.0);
	//solver->solve_equation(GR);
	double flops_factorize = solver->factorize();
	t_factorise = get_time(t_factorise);

	log_det = solver->logDet();

	#ifdef PRINT_MSG
	printf("logdet: %f\n", log_det);
	#endif

	#ifdef PRINT_TIMES
	printf("RGF factorise time: %lg\n",t_factorise);
	#endif

	delete solver;
	delete[] ia;
	delete[] ja;
	delete[] a;


}

void RGFSolver::factorize_solve(SpMat& Q, Vector& rhs, Vector& sol, double &log_det) {

	#ifdef PRINT_MSG
	std::cout << "in RGF FACTORIZE_SOLVE()." << std::endl;
	#endif

	// check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n = %d.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

	// only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 
    nnz = Q_lower.nonZeros();

    size_t* ia; 
    size_t* ja;
    T* a; 

    // allocate memory
    ia = new long unsigned int [n+1];
    ja = new long unsigned int [nnz];
    a = new double [nnz];

    Q_lower.makeCompressed();

    for (i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
    }  

    for (i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
    }  

    for (i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
    }

	double t_factorise;
	double t_solve;
	RGF<T> *solver;

	solver = new RGF<T>(ia, ja, a, ns_t, nt_t, nb_t);

	t_factorise = get_time(0.0);
	//solver->solve_equation(GR);
	double flops_factorize = solver->factorize();
	t_factorise = get_time(t_factorise);

	log_det = solver->logDet();

	#ifdef PRINT_MSG
	printf("logdet: %f\n", log_det);
	#endif

	T *b;
  	T *x;

  	b      = new T[n];
  	x      = new T[n];

  	// assign b to correct format
  	for (i = 0; i < n; i++){
	    b[i] = rhs[i];
	    //printf("%f\n", b[i]);
  	}

  	t_solve = get_time(0.0); 
  	double flops_solve = solver->solve(x, b, 1);
  	t_solve = get_time(t_solve);

  	#ifdef PRINT_MSG
  	//printf("flops solve:     %f\n", flops_solve);
	printf("Residual norm: %e\n", solver->residualNorm(x, b));
	printf("Residual norm normalized: %e\n", solver->residualNormNormalized(x, b));
	#endif

	#ifdef PRINT_TIMES
	printf("RGF factorise time: %lg\n",t_factorise);
  	printf("RGF solve     time: %lg\n",t_solve);
  	#endif

  	// assign b to correct format
  	for (i = 0; i < n; i++){
	    sol[i] = x[i];
  	}	


  	delete solver;

  	delete[] ia;
  	delete[] ja;
  	delete[] a;
  	delete[] x;
  	delete[] b;
}

// IMPLEMENT IN A WAY SUCH THAT FACTORISATION WILL BE PERFORMED AGAIN
// FOR NOW: cannot rely on factorisation to be there.
void RGFSolver::selected_inversion(SpMat& Q, Vector& inv_diag) {

	std::cout << "in RGF SELECTED_INVERSION()." << std::endl;


	#ifdef PRINT_MSG
	std::cout << "in RGF SELECTED_INVERSION()." << std::endl;
	#endif

	// check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n = %d.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

	// only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 
    size_t nnz = Q_lower.nonZeros();

    size_t* ia; 
    size_t* ja;
    T* a; 
  	T* invDiag;

  	invDiag  = new T[n];

    // allocate memory
    ia = new long unsigned int [n+1];
    ja = new long unsigned int [nnz];
    a = new double [nnz];

    Q_lower.makeCompressed();

    for (i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
    }  

    for (i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
    }  

    for (i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
    }

    double t_factorise;
	double t_inv;

	RGF<T> *solver;
	solver = new RGF<T>(ia, ja, a, ns_t, nt_t, nb_t);

	t_factorise = get_time(0.0);
	double flops_factorize = solver->factorize();
	t_factorise = get_time(t_factorise);

  	printf("RGF factorise time: %lg\n",t_factorise);

	t_inv = get_time(0.0);
  	double flops_inv = solver->RGFdiag(invDiag);
  	t_inv = get_time(t_inv);

  	#ifdef PRINT_MSG
  	printf("flops factorise:      %f\n", flops_factorize);
  	printf("flops inv      :      %f\n", flops_inv);
  	#endif

	#ifdef PRINT_TIMES
	printf("RGF factorise time: %lg\n",t_factorise);
  	printf("RGF sel inv time  : %lg\n",t_inv);
  	#endif

  	// fill Eigen vector
  	for (i = 0; i < n; i++){
	    inv_diag[i] = invDiag[i];
  	}


	// free memory
	delete solver;
	delete[] ia;
	delete[] ja;
	delete[] a;
	delete[] invDiag;

}



RGFSolver::~RGFSolver(){
    //std::cout << "Derived destructor called." << std::endl;
}



