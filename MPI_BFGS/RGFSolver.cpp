#include "RGFSolver.h"

#include "RGF/RGF.H"


RGFSolver::RGFSolver(size_t ns_, size_t nt_, size_t nb_, size_t no_) : ns(ns_), nt(nt_), nb(nb_), no(no_){
   	std::cout << "constructing RGF solver." << std::endl;

   	n = ns*nt + nb;

}

// currently not needed !!
void RGFSolver::symbolic_factorization(SpMat& Q, int& init) {
	init = 1;
	std::cout << "Placeholder SYMBOLIC_FACTORIZATION()." << std::endl;
}

void RGFSolver::factorize(SpMat& Q, double& log_det) {
	log_det = 1;
	std::cout << "in RGF FACTORIZE()." << std::endl;

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

    #if 1
	double t_factorise;
	RGF<T> *solver;

	time_t rawtime;
	struct tm *timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	printf ("The current date/time is: %s\n",asctime(timeinfo));

	// cast ns, nt, nb as size_t
	size_t ns_ = ns;
	size_t nt_ = nt;
	size_t nb_ = nb;

	solver = new RGF<T>(ia, ja, a, ns_, nt_, nb_);

	t_factorise = get_time(0.0);
	//solver->solve_equation(GR);
	double flops_factorize = solver->factorize();
	t_factorise = get_time(t_factorise);

	log_det = solver->logDet();
	printf("logdet: %f\n", log_det);

	#endif


}

void RGFSolver::factorize_solve(SpMat& Q, Vector& rhs, Vector& sol, double &log_det) {
	//sol.setOnes();
	//std::cout << "Placeholder FACTORIZE_SOLVE()." << std::endl;

		log_det = 1;
	std::cout << "in RGF FACTORIZE()." << std::endl;

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

    #if 1
	double t_factorise;
	double t_solve;
	RGF<T> *solver;

	time_t rawtime;
	struct tm *timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	printf ("The current date/time is: %s\n",asctime(timeinfo));

	// cast ns, nt, nb as size_t
	size_t ns_ = ns;
	size_t nt_ = nt;
	size_t nb_ = nb;

	solver = new RGF<T>(ia, ja, a, ns_, nt_, nb_);

	t_factorise = get_time(0.0);
	//solver->solve_equation(GR);
	double flops_factorize = solver->factorize();
	t_factorise = get_time(t_factorise);

	log_det = solver->logDet();
	printf("logdet: %f\n", log_det);

	T *b;
  	T *x;

  	b      = new T[n];
  	x      = new T[n];

  	// assign b to correct format
  	for (int i = 0; i < n; i++){
	    b[i] = rhs[i];
	    //printf("%f\n", b[i]);
  	}

  	t_solve = get_time(0.0); 
  	double flops_solve = solver->solve(x, b, 1);
  	t_solve = get_time(t_solve);
  	printf("flops solve:     %f\n", flops_solve);

  	printf("RGF factorise time: %lg\n",t_factorise);
  	printf("RGF solve     time: %lg\n",t_solve);

	printf("Residual norm: %e\n", solver->residualNorm(x, b));
	printf("Residual norm normalized: %e\n", solver->residualNormNormalized(x, b));

  	// assign b to correct format
  	for (int i = 0; i < n; i++){
	    sol[i] = x[i];
	    //printf("%f\n", b[i]);
  	}	

  	#endif
}

void RGFSolver::selected_inversion(SpMat& Q, Vector& inv_diag) {
	inv_diag = 5*Vector::Ones(inv_diag.size());
	std::cout << "Placeholder SELECTED_INVERSION()." << std::endl;
}



RGFSolver::~RGFSolver(){
    //std::cout << "Derived destructor called." << std::endl;
}



