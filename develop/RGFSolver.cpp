
#include "RGFSolver.h"


RGFSolver::RGFSolver(size_t ns, size_t nt, size_t nb, size_t no, int thread_ID_) : ns_t(ns), nt_t(nt), nb_t(nb), no_t(no), thread_ID(thread_ID_){
   	
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);   
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);

#ifdef PRINT_MSG
    std::cout << "constructing RGF solver. MPI rank = " << MPI_rank << ", MPI size = " << MPI_size << std::endl;
#endif

    threads_level1 = omp_get_num_threads();  // get number of threads of current level. not one below.
    //std::cout << "threads level 1 : " << threads_level1 << std::endl;

    // thread_ID only used to set GPU rank & align with appropriate cores
    if(thread_ID-1 > threads_level1 || thread_ID > 1){
        printf("thread_ID = %d, num threads level 1 = %d. max thread_ID = 2! MISMATCH!\n", thread_ID, threads_level1);
        exit(1);
    }

    MPI_Get_processor_name(processor_name, &name_len);
    //printf("Processor name : %s\n",processor_name);


    // CAREFUL USING N in both functions ..
    n  = ns_t*nt_t + nb_t;

    // take out 
    // assign GPU
    int noGPUs;
    cudaGetDeviceCount(&noGPUs);
#ifdef PRINT_MSG
    std::cout << "available GPUs : " << noGPUs << std::endl;
#endif

    if(strcmp ("KW60890", processor_name) == 0){

        GPU_rank = 0;
        if(MPI_rank == 0){
            printf("Careful! GPU rank hard coded to machine: kw60890!\n");
        }

    } else {
        if(MPI_rank == 0){
            printf("assuming I'm on ALEX!\n"); 
        }   

        // assume max 3 ranks per node
        int max_rank_per_node = 4;
        int MPI_rank_mod = MPI_rank % max_rank_per_node; 

        if(MPI_rank_mod == 0){
            //GPU_rank = 2 + get_omp_num_threads(); // GPU2 & 3 attached to NUMA 1
            GPU_rank = 2 + thread_ID; // GPU2 & 3 attached to NUMA 1
        } else if(MPI_rank_mod == 1){
            GPU_rank = 0 + thread_ID; // GPU0 & 1 attached to NUMA 3
        } else if(MPI_rank_mod == 2){
            GPU_rank = 6 + thread_ID; // GPU6 & 7 attached to NUMA 5
        } else if(MPI_rank_mod == 3){
            GPU_rank = 4 + thread_ID; // GPU4 & 5 attached to NUMA 7
        } else {
            printf("too many MPI ranks per node ...\n");
            exit(1);
        }
      
    }
    //GPU_rank = MPI_rank % noGPUs;
    cudaSetDevice(GPU_rank);

    int numa_node = topo_get_numNode(GPU_rank);
    
    int* hwt = NULL;
    int hwt_count = read_numa_threads(numa_node, &hwt);

    // now they will be directly next to each other ... lets see if this is a problem
    pin_hwthreads(1, &hwt[omp_get_thread_num()]);
    std::cout<<"In RGF constructor. nb = "<<nb<<", MPI rank: "<<MPI_rank<< ", hostname: "<<processor_name<<", GPU rank : "<<GPU_rank <<", tid: "<<omp_get_thread_num()<<", NUMA domain ID: "<<numa_node;
    std::cout<<", hwthreads: " << hwt[omp_get_thread_num()] << std::endl;

#ifdef PRINT_MSG
    std::cout << "RGF constructor, nb = " << nb << ", MPI rank : " << MPI_rank << ", hostname : " << processor_name << ", GPU rank : " << GPU_rank << std::endl;
#endif	
    
    exit(1);
    solver = new RGF<double>(ns_t, nt_t, nb_t);
 
#ifdef PRINT_MSG    
    if(MPI_rank == 0)
    	std::cout << "new RGFSolver Class version." << std::endl; 
#endif

}

// currently not needed !!
void RGFSolver::symbolic_factorization(SpMat& Q, int& init) {
	init = 1;
	std::cout << "Placeholder SYMBOLIC_FACTORIZATION() not needed for RGF." << std::endl;
}

// NOTE: this function is written to factorize prior! Assumes tridiagonal structure.
void RGFSolver::factorize(SpMat& Q, double& log_det, double& t_priorLatChol) {

#ifdef PRINT_MSG
	std::cout << "MPI rank : " << MPI_rank << ", in RGF FACTORIZE()." << std::endl;
#endif

    // check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n-nb = %ld.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

#ifdef PRINT_MSG
    	std::cout << "Q in RGFSolver.cpp : \n" << Q.block(0,0,10,10) << std::endl;
#endif

	// only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 
    nnz = Q_lower.nonZeros();

    size_t* ia = new long unsigned int [n+1];
    size_t* ja = new long unsigned int [nnz];
    double* a = new double [nnz];

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

#ifdef PRINT_MSG
    printf("Calling RGF solver in RGF factorize now.\n");
#endif

       // std::cout << "RGF factorzie, nb = " << nb_t << " , MPI rank : " << MPI_rank << ", tid : " << omp_get_thread_num() << ", GPU rank : " << GPU_rank << std::endl;

    t_priorLatChol = get_time(0.0);
    double gflops_factorize = solver->factorize_noCopyHost(ia, ja, a, log_det);
    //std::cout << "log_det new      = " << log_det << std::endl;

    //double gflops_factorize = solver->factorize();
    //log_det = solver->logDet();
    //std::cout << "log_det original = " << log_det << std::endl;
    t_priorLatChol = get_time(t_priorLatChol);

#ifdef GFLOPS
    if(MPI_rank == 0){
        std::cout << "Gflop/s for the numerical factorization Qu: " << gflops_factorize << std::endl;
    }
#endif

     //std::cout << "In factorize. hostname : " << processor_name << ", MPI_rank : " << MPI_rank << ", GPU rank : " << GPU_rank << ", time Chol : " << t_priorLatChol << std::endl;

#ifdef PRINT_MSG
	printf("logdet: %f\n", log_det);
#endif

#ifdef PRINT_TIMES
	printf("RGF factorise time: %lg\n",t_priorLatChol);
#endif

	delete[] ia;
	delete[] ja;
	delete[] a;
}

void RGFSolver::factorize_w_constr(SpMat& Q, const MatrixXd& D, double& log_det, MatrixXd& V){

    int nrhs = D.rows();

#ifdef PRINT_MSG
    std::cout << "in RGF FACTORIZE W CONSTRAINTS()." << std::endl;
#endif
    
    // check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n-nb = %ld.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

#ifdef PRINT_MSG
        std::cout << "Q in RGFSolver.cpp : \n" << Q.block(0,0,10,10) << std::endl;
#endif

    // only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 
    nnz = Q_lower.nonZeros();

    // allocate memory
    size_t* ia = new long unsigned int [n+1];
    size_t* ja = new long unsigned int [nnz];
    double* a = new double [nnz];

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

#ifdef PRINT_MSG
    printf("Calling RGF solver in RGF factorize w constraints now.\n");
#endif

    double t_factorise = get_time(0.0);
    double gflops_factorize = solver->factorize(ia, ja, a);
    t_factorise = get_time(t_factorise);

    log_det = solver->logDet(ia, ja, a);

#ifdef PRINT_MSG
    printf("logdet: %f\n", log_det);
    if(isnan(log_det))
	    exit(1);
#endif

#ifdef PRINT_TIMES
    printf("RGF factorise time: %lg\n",t_factorise);
#endif

    // ============== solve for constraints now =========== //
    double* b      = new double[n*nrhs];
    double* x      = new double[n*nrhs];

    // assign b to correct format
    // want rows of D to become vector ... 
    MatrixXd Dt = D.transpose();
    memcpy(b, Dt.data(), n*nrhs*sizeof(double));

    double t_solve = get_time(0.0); 
    double gflops_solve = solver->solve(ia, ja, a, x, b, nrhs);
    t_solve = get_time(t_solve);

#ifdef PRINT_MSG
    //printf("flops solve:     %f\n", flops_solve);
    printf("Residual norm: %e\n", solver->residualNorm(x, b));
    printf("Residual norm normalized: %e\n", solver->residualNormNormalized(x, b));
#endif

    // map solution back
    memcpy(V.data(), x, nrhs*n*sizeof(double));
    //std::cout << "norm(Qst*V   - t(Dx)) = " << (Q*V - D.transpose()).norm() << ", norm(V) = " << V.norm() << std::endl;


#ifdef PRINT_TIMES
    printf("RGF factorise time: %lg\n",t_factorise);
    printf("RGF solve     time: %lg\n",t_solve);
#endif

    delete[] ia;
    delete[] ja;
    delete[] a;

    delete [] b;
    delete [] x;

}  // end factorize w constraints

void RGFSolver::factorize_solve(SpMat& Q, Vect& rhs, Vect& sol, double &log_det, double& t_condLatChol, double& t_condLatSolve) {

    int nrhs = 1;

#ifdef PRINT_MSG
    std::cout << "MPI rank : " << MPI_rank << ", in RGF FACTORIZE_SOLVE()." << std::endl;	
#endif

    // check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n = %ld.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

	// only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 
    nnz = Q_lower.nonZeros();

#ifdef PRINT_MSG
    std::cout << "nnz Q = " << nnz << std::endl;
#endif

     // pin here

    // allocate memory
    size_t* ia = new long unsigned int [n+1];
    size_t* ja = new long unsigned int [nnz];
    double* a = new double [nnz];

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

#ifdef PRINT_MSG
	std::cout << "calling solver = new RGF now. ns = " << ns_t << ", nt = " << nt_t << ", nb = " << nb_t << std::endl;
#endif


#ifdef PRINT_MSG
    printf("Calling RGF solver in RGF factorize_solver now.\n");
#endif

       // std::cout << "RGF factorize solve, nb = " << nb_t << ", MPI rank : " << MPI_rank << ", tid : " << omp_get_thread_num() << ", GPU rank : " << GPU_rank << std::endl;

    t_condLatChol = get_time(0.0);

	double gflops_factorize = solver->factorize(ia, ja, a);
    //double gflops_factorize = solver->factorize();

    t_condLatChol = get_time(t_condLatChol);
    
	log_det = solver->logDet(ia, ja, a);
    //log_det = solver->logDet();

#ifdef GFLOPS
    if(MPI_rank == 0){
        std::cout << "Gflop/s for the numerical factorization Qxy: " << gflops_factorize << std::endl;
    }
#endif
    

#ifdef PRINT_MSG
	printf("logdet: %f\n", log_det);
#endif


  	double* b      = new double[n];
  	double* x      = new double[n];

  	// assign b to correct format
  	for (i = 0; i < n; i++){
	    b[i] = rhs[i];
	    //printf("%f\n", b[i]);
  	}

    t_condLatSolve = get_time(0.0);

  	double gflops_solve = solver->solve(ia, ja, a, x, b, nrhs);
    //double gflops_solve = solver->solve(x, b, nrhs);

    t_condLatSolve = get_time(t_condLatSolve);

#ifdef PRINT_MSG
  	//printf("flops solve:     %f\n", flops_solve);
	printf("Residual norm: %e\n", solver->residualNorm(x, b));
	printf("Residual norm normalized: %e\n", solver->residualNormNormalized(x, b));
#endif

#ifdef PRINT_TIMES
	printf("RGF factorise time: %lg\n",t_condLatChol);
  	printf("RGF solve     time: %lg\n",t_condLatSolve);
#endif

	//std::cout << "In factorize_solve. hostname : " << processor_name << ", MPI_rank : " << MPI_rank << ", GPU rank : " << GPU_rank << ", time Chol : " << t_condLatChol << ", time Solve : " << t_condLatSolve << std::endl;

  	// assign b to correct format
  	for (i = 0; i < n; i++){
	    sol[i] = x[i];
  	}	


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

    int nrhs = Dxy.rows() + 1;
    
    // check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n = %ld.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

    // only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 
    nnz = Q_lower.nonZeros();

#ifdef PRINT_MSG
    std::cout << "nnz Qxy = " << nnz << std::endl;
#endif

    // allocate memory
    size_t* ia = new long unsigned int [n+1];
    size_t* ja = new long unsigned int [nnz];
    double* a = new double [nnz];

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

#ifdef PRINT_MSG
    std::cout << "Calling solver = new RGF() now" << std::endl;
#endif

#ifdef PRINT_MSG
    printf("Calling RGF solver in RGF factorize_solver now.\n");
#endif

    double t_factorise = get_time(0.0);
    //solver->solve_equation(GR);
    double gflops_factorize = solver->factorize(ia, ja, a);
    t_factorise = get_time(t_factorise);

    log_det = solver->logDet(ia, ja, a);

#ifdef PRINT_MSG
    printf("logdet: %f\n", log_det);
#endif

    double* b      = new double[n*nrhs];
    double* x      = new double[n*nrhs];

    memcpy(b, rhs.data(), n*sizeof(double));
    
    //std::cout << "in constr is true" << std::endl;
    MatrixXd Dt = Dxy.transpose();
    // Dxy.transpose().data() is not sufficient ... 
    memcpy(b + n, Dt.data(), n*Dxy.rows()*sizeof(double));

    double t_solve = get_time(0.0); 
    double gflops_solve = solver->solve(ia, ja, a, x, b, nrhs);
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

    delete[] ia;
    delete[] ja;
    delete[] a;
    delete[] x;
    delete[] b;

} // end factorize solve with constraints


// IMPLEMENT IN A WAY SUCH THAT FACTORISATION WILL BE PERFORMED AGAIN
// FOR NOW: cannot rely on factorisation to be there.
void RGFSolver::selected_inversion(SpMat& Q, Vect& inv_diag) {

#ifdef PRINT_MSG
    std::cout << "in RGF SELECTED_INVERSION()." << std::endl;
#endif

    // check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n = %ld.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

    // only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 
    size_t nnz = Q_lower.nonZeros();

    // allocate memory
    size_t* ia = new long unsigned int [n+1];
    size_t* ja = new long unsigned int [nnz];
    double* a = new double [nnz];

    double* invDiag  = new double[n];

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

    t_factorise = get_time(0.0);
    double gflops_factorize = solver->factorize(ia, ja, a);
    t_factorise = get_time(t_factorise);

#ifdef PRINT_TIMES
    printf("RGF factorise time: %lg\n",t_factorise);
#endif

    t_inv = get_time(0.0);
    double gflops_inv = solver->RGFdiag(ia, ja, a, invDiag);
    t_inv = get_time(t_inv);

#ifdef PRINT_MSG
    printf("gflops factorise:      %f\n", gflops_factorize);
    printf("gflops inv      :      %f\n", gflops_inv);
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
    delete[] ia;
    delete[] ja;
    delete[] a;
    delete[] invDiag;

} // end selected inversion function


void RGFSolver::selected_inversion_w_constr(SpMat& Q, const MatrixXd& D, Vect& inv_diag, MatrixXd& V){

    int nrhs = D.rows();

#ifdef PRINT_MSG
    std::cout << "in RGF SELECTED_INVERSION_W_CONSTR()." << std::endl;
#endif

    // check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n = %ld.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

    // only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 
    size_t nnz = Q_lower.nonZeros();

    // allocate memory
    size_t* ia = new long unsigned int [n+1];
    size_t* ja = new long unsigned int [nnz];
    double* a = new double [nnz];

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

    double* invDiag  = new double[n];
    // for Q*V = D
    double* b = new double[nrhs*n];
    double* x = new double[nrhs*n];

    // MatrixXd regularly in column-major, hence
    MatrixXd Dt = D.transpose();
    // Dxy.transpose().data() is not sufficient ... 
    memcpy(b, Dt.data(), n*nrhs*sizeof(double));

    double t_factorise, t_solve, t_inv;

    t_factorise = get_time(0.0);
    double flops_factorize = solver->factorize(ia, ja, a);
    t_factorise = get_time(t_factorise);

#ifdef PRINT_TIMES
    printf("RGF factorise time: %lg\n",t_factorise);
#endif

    t_solve = get_time(0.0); 
    double flops_solve = solver->solve(ia, ja, a, x, b, nrhs);
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
    double flops_inv = solver->RGFdiag(ia, ja, a, invDiag);
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
    delete[] ia;
    delete[] ja;
    delete[] a;

    delete[] b;
    delete[] x;

    delete[] invDiag;

}  // end selected inversion with constraints


RGFSolver::~RGFSolver(){
    //std::cout << "Derived destructor called." << std::endl;
    delete solver;

}




