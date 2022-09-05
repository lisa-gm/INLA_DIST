#include "RGFSolver.h"

#include "../RGF/RGF.H"


RGFSolver::RGFSolver(size_t ns, size_t nt, size_t nb, size_t no) : ns_t(ns), nt_t(nt), nb_t(nb), no_t(no){
   	
#ifdef PRINT_MSG
   	std::cout << "constructing RGF solver." << std::endl;
#endif

   	MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);   
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);

    threads_level1 = omp_get_max_threads();
    //std::cout << "threads level 1 : " << threads_level1 << std::endl;

    // CAREFUL USING N in both functions ..
   	//n  = ns_t*nt_t + nb_t;

}

// currently not needed !!
void RGFSolver::symbolic_factorization(SpMat& Q, int& init) {
	init = 1;
	std::cout << "Placeholder SYMBOLIC_FACTORIZATION() not needed for RGF." << std::endl;
}

// NOTE: this function is written to factorize prior! Assumes tridiagonal structure.
void RGFSolver::factorize(SpMat& Q, double& log_det, double& t_priorLatChol) {

    unsigned int n = ns_t*nt_t;

#ifdef PRINT_MSG
	std::cout << "in RGF FACTORIZE()." << std::endl;
#endif

	// assign GPU
    int noGPUs; // = 2;
    cudaGetDeviceCount(&noGPUs);
#ifdef PRINT_MSG
    std::cout << "available GPUs : " << noGPUs << std::endl;
#endif
    // allocate devices as numThreads mod noGPUs
    int counter = threads_level1*MPI_rank + omp_get_thread_num();
    int GPU_rank = counter % noGPUs;
    cudaSetDevice(GPU_rank);
//#ifdef PRINT_MSG
    std::cout << "FACT -- counter : " << counter << ", MPI rank : " << MPI_rank << ", tid : " << omp_get_thread_num() << ", GPU rank : " << GPU_rank << std::endl;
//#endif

    // check if n and Q.size() match
    if(n != Q.rows()){
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

#ifdef PRINT_MSG
    printf("Calling RGF solver in RGF factorize now.\n");
#endif

	t_factorise = get_time(0.0);
	//solver->solve_equation(GR);

    t_priorLatChol = get_time(0.0);
    double gflops_factorize = solver->factorize_noCopyHost(log_det);
    //std::cout << "log_det new      = " << log_det << std::endl;
	
    //double flops_factorize = solver->factorize();
    //log_det = solver->logDet();
    //std::cout << "log_det original = " << log_det << std::endl;
    t_priorLatChol = get_time(t_priorLatChol);

	t_factorise = get_time(t_factorise);

    if(MPI_rank == 0){
        std::cout << "Gflop/s for the numerical factorization: " << gflops_factorize << std::endl;
    }

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


void RGFSolver::factorize_w_constr(SpMat& Q, const MatrixXd& D, double& log_det, MatrixXd& V){

    int nrhs = D.rows();
    unsigned int n = ns_t*nt_t;

#ifdef PRINT_MSG
    std::cout << "in RGF FACTORIZE W CONSTRAINTS()." << std::endl;
#endif

    // assign GPU
    int noGPUs; 
    cudaGetDeviceCount(&noGPUs);
#ifdef PRINT_MSG
    std::cout << "available GPUs : " << noGPUs << std::endl;
#endif
    // allocate devices as numThreads mod noGPUs
    int counter = threads_level1*MPI_rank + omp_get_thread_num();
    int GPU_rank = counter % noGPUs;
    cudaSetDevice(GPU_rank);
#ifdef PRINT_MSG
    std::cout << "FACT -- counter : " << counter << ", MPI rank : " << MPI_rank << ", tid : " << omp_get_thread_num() << ", GPU rank : " << GPU_rank << std::endl;
#endif

    // check if n and Q.size() match
    if(n != Q.rows()){
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

    double t_factorise, t_solve;
    RGF<T> *solver;

    // SET nb_t to zero : this function is for precision matrix of prior!
    solver = new RGF<T>(ia, ja, a, ns_t, nt_t, 0);

#ifdef PRINT_MSG
    printf("Calling RGF solver in RGF factorize w constraints now.\n");
#endif

    t_factorise = get_time(0.0);
    //solver->solve_equation(GR);
    double gflops_factorize = solver->factorize();
    t_factorise = get_time(t_factorise);

    log_det = solver->logDet();

#ifdef PRINT_MSG
    printf("logdet: %f\n", log_det);
#endif

#ifdef PRINT_TIMES
    printf("RGF factorise time: %lg\n",t_factorise);
#endif

    // ============== solve for constraints now =========== //
    T *b;
    T *x;

    b      = new T[n*nrhs];
    x      = new T[n*nrhs];

    // assign b to correct format
    // want rows of D to become vector ... 
    MatrixXd Dt = D.transpose();
    memcpy(b, Dt.data(), n*nrhs*sizeof(double));

    t_solve = get_time(0.0); 
    double gflops_solve = solver->solve(x, b, nrhs);
    t_solve = get_time(t_solve);

#ifdef PRINT_MSG
    //printf("flops solve:     %f\n", flops_solve);
    printf("Residual norm: %e\n", solver->residualNorm(x, b));
    printf("Residual norm normalized: %e\n", solver->residualNormNormalized(x, b));
#endif

    // map solution back
    memcpy(V.data(), x, nrhs*n*sizeof(double));
    //std::cout << "norm(Qst*V   - t(Dx)) = " << (Q*V - D.transpose()).norm() << std::endl;


#ifdef PRINT_TIMES
    printf("RGF factorise time: %lg\n",t_factorise);
    printf("RGF solve     time: %lg\n",t_solve);
#endif

    delete solver;
    delete[] ia;
    delete[] ja;
    delete[] a;

    delete [] b;
    delete [] x;

}  // end factorize w constraints

void RGFSolver::factorize_solve(SpMat& Q, Vect& rhs, Vect& sol, double &log_det, double& t_condLatChol, double& t_condLatSolve) {

    int nrhs = 1;
    unsigned int n = ns_t*nt_t + nb_t;

#ifdef PRINT_MSG
	std::cout << "in RGF FACTORIZE_SOLVE()." << std::endl;
#endif

	// assign GPU
    int noGPUs;
    cudaGetDeviceCount(&noGPUs);
#ifdef PRINT_MSG
    std::cout << "available GPUs : " << noGPUs << std::endl;
#endif
    // allocate devices as numThreads mod noGPUs
    int tid = omp_get_thread_num();
    int counter = threads_level1*MPI_rank + tid;
	int GPU_rank = counter % noGPUs;
    cudaSetDevice(GPU_rank);
//#ifdef PRINT_MSG
    std::cout << "FACT & SOLVE -- counter : " << counter << ", MPI rank : " << MPI_rank  << ", tid : " << tid << ", GPU rank : " << GPU_rank << std::endl;
//#endif

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

#ifdef PRINT_MSG
    printf("Calling RGF solver in RGF factorize_solver now.\n");
#endif

	t_factorise = get_time(0.0);
    t_condLatChol = get_time(0.0);
	//solver->solve_equation(GR);
	double gflops_factorize = solver->factorize();
    t_condLatChol = get_time(t_condLatChol);
	t_factorise = get_time(t_factorise);
    
	log_det = solver->logDet();

    /*
    if(MPI_rank == 0){
        std::cout << "Gflop/s for the numerical factorization: " << gflops_factorize << std::endl;
    }
    */

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
    t_condLatSolve = get_time(0.0);

  	double gflops_solve = solver->solve(x, b, nrhs);
    
    t_condLatSolve = get_time(t_condLatSolve);
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

} // factorize solve

void RGFSolver::factorize_solve_w_constr(SpMat& Q, Vect& rhs, const MatrixXd& Dxy, double &log_det, Vect& sol, MatrixXd& V){

#ifdef PRINT_MSG
    std::cout << "in RGF FACTORIZE_SOLVE_W_CONSTR()." << std::endl;
#endif

    unsigned int n = ns_t*nt_t + nb_t;
    int nrhs = Dxy.rows() + 1;

    // assign GPU
    int noGPUs;
    cudaGetDeviceCount(&noGPUs);
#ifdef PRINT_MSG
    std::cout << "available GPUs : " << noGPUs << std::endl;
#endif
    // allocate devices as numThreads mod noGPUs
    int tid = omp_get_thread_num();
    int counter = threads_level1*MPI_rank + tid;
    int GPU_rank = counter % noGPUs;
    cudaSetDevice(GPU_rank);
#ifdef PRINT_MSG
    std::cout << "FACT & SOLVE -- counter : " << counter << ", MPI rank : " << MPI_rank  << ", tid : " << tid << ", GPU rank : " << GPU_rank << std::endl;
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

#ifdef PRINT_MSG
    printf("Calling RGF solver in RGF factorize_solver now.\n");
#endif

    t_factorise = get_time(0.0);
    //solver->solve_equation(GR);
    double gflops_factorize = solver->factorize();
    t_factorise = get_time(t_factorise);

    log_det = solver->logDet();

#ifdef PRINT_MSG
    printf("logdet: %f\n", log_det);
#endif

    T *b;
    T *x;

    b      = new T[n*nrhs];
    x      = new T[n*nrhs];

    memcpy(b, rhs.data(), n*sizeof(double));
    
    //std::cout << "in constr is true" << std::endl;
    MatrixXd Dt = Dxy.transpose();
    // Dxy.transpose().data() is not sufficient ... 
    memcpy(b + n, Dt.data(), n*Dxy.rows()*sizeof(double));

    t_solve = get_time(0.0); 
    double gflops_solve = solver->solve(x, b, nrhs);
    t_solve = get_time(t_solve);

#ifdef PRINT_MSG
    //printf("flops solve:     %f\n", flops_solve);
    printf("Residual norm: %e\n", solver->residualNorm(x, b));
    printf("Residual norm normalized: %e\n", solver->residualNormNormalized(x, b));
#endif

    // map solution back
    memcpy(sol.data(), x, n*sizeof(double));
    //std::cout << "norm(Q*sol - rhs) = " << (Q*sol - rhs).norm() << std::endl;

    memcpy(V.data(), x+n, n*Dxy.rows()*sizeof(double));
    //std::cout << "norm(Q*V   - t(Dxy) = " << (Q*V   - Dxy.transpose()).norm() << std::endl;

#ifdef PRINT_TIMES
    printf("RGF factorise time: %lg\n",t_factorise);
    printf("RGF solve     time: %lg\n",t_solve);
#endif

    delete solver;

    delete[] ia;
    delete[] ja;
    delete[] a;
    delete[] x;
    delete[] b;

} // end factorize solve with constraints

// IMPLEMENT IN A WAY SUCH THAT FACTORISATION WILL BE PERFORMED AGAIN
// FOR NOW: cannot rely on factorisation to be there.
void RGFSolver::selected_inversion(SpMat& Q, Vect& inv_diag) {

    unsigned int n = ns_t*nt_t + nb_t;

#ifdef PRINT_MSG
	std::cout << "in RGF SELECTED_INVERSION()." << std::endl;
#endif

	// assign GPU
    int noGPUs;
    cudaGetDeviceCount(&noGPUs);
#ifdef PRINT_MSG
    std::cout << "available GPUs : " << noGPUs << std::endl;
#endif
    // allocate devices as numThreads mod noGPUs
    int tid = omp_get_thread_num();
    int counter = threads_level1*MPI_rank + tid;
	int GPU_rank = counter % noGPUs;
    cudaSetDevice(GPU_rank);
#ifdef PRINT_MSG
    std::cout << "counter : " << counter << ", MPI rank : " << MPI_rank  << ", tid : " << tid << ", GPU rank : " << GPU_rank << std::endl;
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

    double t_factorise, t_inv;

	RGF<T> *solver;
	solver = new RGF<T>(ia, ja, a, ns_t, nt_t, nb_t);

	t_factorise = get_time(0.0);
	double gflops_factorize = solver->factorize();
	t_factorise = get_time(t_factorise);

#ifdef PRINT_TIMES
  	printf("RGF factorise time: %lg\n",t_factorise);
#endif

	t_inv = get_time(0.0);
  	double gflops_inv = solver->RGFdiag(invDiag);
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

} // end selected inversion function


void RGFSolver::selected_inversion_w_constr(SpMat& Q, const MatrixXd& D, Vect& inv_diag, MatrixXd& V){

    unsigned int n = ns_t*nt_t + nb_t;
    int nrhs = D.rows();


#ifdef PRINT_MSG
    std::cout << "in RGF SELECTED_INVERSION_W_CONSTR()." << std::endl;
#endif

    // assign GPU
    int noGPUs;
    cudaGetDeviceCount(&noGPUs);
#ifdef PRINT_MSG
    std::cout << "available GPUs : " << noGPUs << std::endl;
#endif
    // allocate devices as numThreads mod noGPUs
    int tid = omp_get_thread_num();
    int counter = threads_level1*MPI_rank + tid;
    int GPU_rank = counter % noGPUs;
    cudaSetDevice(GPU_rank);
#ifdef PRINT_MSG
    std::cout << "counter : " << counter << ", MPI rank : " << MPI_rank  << ", tid : " << tid << ", GPU rank : " << GPU_rank << std::endl;
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
    T* b;
    T* x;

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

    // for Q*V = D
    b = new T[nrhs*n];
    x = new T[nrhs*n];

    // MatrixXd regularly in column-major, hence
    MatrixXd Dt = D.transpose();
    // Dxy.transpose().data() is not sufficient ... 
    memcpy(b, Dt.data(), n*nrhs*sizeof(double));

    double t_factorise, t_solve, t_inv;

    RGF<T> *solver;
    solver = new RGF<T>(ia, ja, a, ns_t, nt_t, nb_t);

    t_factorise = get_time(0.0);
    double flops_factorize = solver->factorize();
    t_factorise = get_time(t_factorise);

#ifdef PRINT_TIMES
    printf("RGF factorise time: %lg\n",t_factorise);
#endif

    t_solve = get_time(0.0); 
    double flops_solve = solver->solve(x, b, nrhs);
    t_solve = get_time(t_solve);

//#ifdef PRINT_MSG
    //printf("flops solve:     %f\n", flops_solve);
    printf("Residual norm: %e\n", solver->residualNorm(x, b));
    printf("Residual norm normalized: %e\n", solver->residualNormNormalized(x, b));
//#endif

    // map solution back
    // V has column-major format
    memcpy(V.data(), x, nrhs*n*sizeof(double));    

    t_inv = get_time(0.0);
    double flops_inv = solver->RGFdiag(invDiag);
    t_inv = get_time(t_inv);

#ifdef PRINT_MSG
    printf("flops factorise:      %f\n", flops_factorize);
    printf("flops solve    :      %f\n", flops_solve);
    printf("flops inv      :      %f\n", flops_inv);
#endif

#ifdef PRINT_TIMES
    printf("RGF factorise time: %lg\n",t_factorise);
    printf("RGF solve time    : %lg\n",t_solve);
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

    delete[] b;
    delete[] x;

    delete[] invDiag;

}  // end selected inversion with constraints




RGFSolver::~RGFSolver(){
    //std::cout << "Derived destructor called." << std::endl;
}



