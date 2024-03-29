
#include "BTASolver.h"


BTASolver::BTASolver(size_t ns, size_t nt, size_t nb) : ns_t(ns), nt_t(nt), nb_t(nb){
   	

    MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);   
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);

#ifdef PRINT_MSG
    std::cout << "constructing BTA solver. MPI rank = " << MPI_rank << ", MPI size = " << MPI_size << std::endl;
#endif

    threads_level1 = omp_get_num_threads();  // get number of threads of current level. not one below.
    //std::cout << "threads level 1 : " << threads_level1 << std::endl;

    MPI_Get_processor_name(processor_name, &name_len);

    // CAREFUL USING N in both functions ..
    n  = ns_t*nt_t + nb_t;

    // take out 
    // assign GPU
    int noGPUs;
    cudaGetDeviceCount(&noGPUs);
#ifdef PRINT_MSG
    std::cout << "available GPUs : " << noGPUs << std::endl;
#endif


    // assume max 3 ranks per node
    int max_rank_per_node = 4;
    int MPI_rank_mod = MPI_rank % max_rank_per_node; 

    if(MPI_rank_mod == 0){
	   GPU_rank = 2 + omp_get_thread_num(); // GPU2 & 3 attached to NUMA 1
    } else if(MPI_rank_mod == 1){
	   GPU_rank = 0 + omp_get_thread_num(); // GPU0 & 1 attached to NUMA 3
    } else if(MPI_rank_mod == 2){
	   GPU_rank = 6 + omp_get_thread_num(); // GPU6 & 7 attached to NUMA 5
    } else if(MPI_rank_mod == 3){
        GPU_rank = 4 + omp_get_thread_num(); // GPU4 & 5 attached to NUMA 7
    } else {
       printf("too many MPI ranks per node ...\n");
       exit(1);
    } 

    
    //int max_rank_per_node = 2;
    //int GPU_rank = 0;   // MPI_rank % max_rank_per_node;

    // allocate devices as numThreads mod noGPUs
    //int counter = threads_level1*MPI_rank + omp_get_thread_num();
    //std::cout << "omp get nested : " << omp_get_nested() << std::endl;
/*
    if(omp_get_nested() == 1 || threads_level1 == 2){
	//int counter = 4;
	// assume that not more than 3 ranks per node ... mod 3
	// first 3 ranks on first node, second 3 on second, etc
	int counter = 2*(MPI_rank % 3) + omp_get_thread_num(); // test shift by 1 ... to not be on the same NUMA domains ...
	GPU_rank = counter % noGPUs;
	//std::cout << "BTA constructor, nb = " << nb << ", MPI rank : " << MPI_rank << ", hostname : " << processor_name << ", GPU rank : " << GPU_rank << ", counter : " << counter << ", tid : " << omp_get_thread_num() << std::endl;
    } else {
	int counter = 2*(MPI_rank % 3) + omp_get_thread_num();  
        GPU_rank = counter % noGPUs;
	//std::cout << "omp nested false. In BTA constructor, nb = " << nb << ", MPI rank : " << MPI_rank << ", hostname : " << processor_name << ", GPU rank : " << GPU_rank << std::endl;
    }
*/    
    //GPU_rank = MPI_rank % noGPUs;
    cudaSetDevice(GPU_rank);

    int numa_node = topo_get_numNode(GPU_rank);
    
    //int numa_node = GPU_rank;
    /*
     int numa_node;

    if(GPU_rank == 0){
	numa_node = 2;
    } else if(GPU_rank == 1){
	numa_node = 3;
    } else if(GPU_rank == 2){
	numa_node = 0;
    } else if(GPU_rank == 3){
	numa_node = 1;
    } else if(GPU_rank == 4){
       numa_node = 6;
    } else if(GPU_rank == 5){
       numa_node = 7;
    } else if(GPU_rank == 6){
       numa_node = 4;
    } else if(GPU_rank == 7){
       numa_node = 5;
    }
    */

    int* hwt = NULL;
    int hwt_count = read_numa_threads(numa_node, &hwt);

    /*pin_hwthreads(hwt_count, hwt);
    std::cout<<"In BTA constructor. nb = "<<nb<<", MPI rank: "<<MPI_rank<< ", hostname: "<<processor_name<<", GPU rank : "<<GPU_rank <<", tid: "<<omp_get_thread_num()<<", NUMA domain ID: "<<numa_node;
    std::cout << ", hwthreads:";
    for(int i=0; i<hwt_count; i++){
        printf(" %d", hwt[i]);
    }
    printf("\n");
    */

    // now they will be directly next to each other ... lets see if this is a problem
    //pin_hwthreads(1, &hwt[omp_get_thread_num()]);
    //std::cout<<"In BTA constructor. nb = "<<nb<<", MPI rank: "<<MPI_rank<< ", hostname: "<<processor_name<<", GPU rank : "<<GPU_rank <<", tid: "<<omp_get_thread_num()<<", NUMA domain ID: "<<numa_node;
    //std::cout<<", hwthreads: " << hwt[omp_get_thread_num()] << std::endl;

#ifdef PRINT_MSG
    std::cout << "BTA constructor, nb = " << nb << ", MPI rank : " << MPI_rank << ", hostname : " << processor_name << ", GPU rank : " << GPU_rank << std::endl;
#endif	
    
    solver = new BTA<double>(ns_t, nt_t, nb_t, MPI_rank);
 
#ifdef PRINT_MSG    
    if(MPI_rank == 0)
    	std::cout << "new BTASolver Class version." << std::endl; 
#endif

}

// currently not needed !!
void BTASolver::symbolic_factorization(SpMat& Q, int& init) {
	init = 1;
	std::cout << "Placeholder SYMBOLIC_FACTORIZATION() not needed for BTA." << std::endl;
}

// NOTE: this function is written to factorize prior! Assumes tridiagonal structure.
void BTASolver::factorize(SpMat& Q, double& log_det, double& t_priorLatChol) {

#ifdef PRINT_MSG
	std::cout << "MPI rank : " << MPI_rank << ", in BTA FACTORIZE()." << std::endl;
#endif


    // check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n-nb = %ld.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

#ifdef PRINT_MSG
    	std::cout << "Q in BTASolver.cpp : \n" << Q.block(0,0,10,10) << std::endl;
#endif

	// only take lower triangular part of A
    SpMat Q_lower = Q.triangularView<Lower>(); 
    nnz = Q_lower.nonZeros();

    // allocate memory
    
    // pin here!

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
    printf("Calling BTA solver in BTA factorize now.\n");
#endif

       // std::cout << "BTA factorzie, nb = " << nb_t << " , MPI rank : " << MPI_rank << ", tid : " << omp_get_thread_num() << ", GPU rank : " << GPU_rank << std::endl;

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
	printf("BTA factorise time: %lg\n",t_priorLatChol);
#endif

	delete[] ia;
	delete[] ja;
	delete[] a;
}

void BTASolver::factorize_w_constr(SpMat& Q, const MatrixXd& D, double& log_det, MatrixXd& V){

    int nrhs = D.rows();

#ifdef PRINT_MSG
    std::cout << "in BTA FACTORIZE W CONSTRAINTS()." << std::endl;
#endif
    
    // check if n and Q.size() match
    if(n != Q.rows()){
        printf("\nInitialised matrix size and current matrix size don't match!\n");
        printf("n-nb = %ld.\nnrows(Q) = %ld.\n", n, Q.rows());
        exit(1);
    }

#ifdef PRINT_MSG
        std::cout << "Q in BTASolver.cpp : \n" << Q.block(0,0,10,10) << std::endl;
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
    printf("Calling BTA solver in BTA factorize w constraints now.\n");
#endif

    double t_factorise = get_time(0.0);
    double gflops_factorize = solver->factorize(ia, ja, a, dummy_time_1);
    t_factorise = get_time(t_factorise);

    log_det = solver->logDet(ia, ja, a);

#ifdef PRINT_MSG
    printf("logdet: %f\n", log_det);
    if(isnan(log_det))
	    exit(1);
#endif

#ifdef PRINT_TIMES
    printf("BTA factorise time: %lg\n",t_factorise);
#endif

    // ============== solve for constraints now =========== //
    double* b      = new double[n*nrhs];
    double* x      = new double[n*nrhs];

    // assign b to correct format
    // want rows of D to become vector ... 
    MatrixXd Dt = D.transpose();
    memcpy(b, Dt.data(), n*nrhs*sizeof(double));

    double t_solve = get_time(0.0); 
    double gflops_solve = solver->solve(ia, ja, a, x, b, nrhs, dummy_time_1, dummy_time_2);
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
    printf("BTA factorise time: %lg\n",t_factorise);
    printf("BTA solve     time: %lg\n",t_solve);
#endif

    delete[] ia;
    delete[] ja;
    delete[] a;

    delete [] b;
    delete [] x;

}  // end factorize w constraints

void BTASolver::factorize_solve(SpMat& Q, Vect& rhs, Vect& sol, double &log_det, double& t_condLatChol, double& t_condLatSolve) {

    int nrhs = 1;

#ifdef PRINT_MSG
    std::cout << "MPI rank : " << MPI_rank << ", in BTA FACTORIZE_SOLVE()." << std::endl;	
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
	std::cout << "calling solver = new BTA now" << std::endl;
#endif

	//solver = new BTA<double>(ia, ja, a, ns_t, nt_t, nb_t);
    //solver = new BTA<double>(ns_t, nt_t, nb_t);


#ifdef PRINT_MSG
    printf("Calling BTA solver in BTA factorize_solver now.\n");
#endif

       // std::cout << "BTA factorize solve, nb = " << nb_t << ", MPI rank : " << MPI_rank << ", tid : " << omp_get_thread_num() << ", GPU rank : " << GPU_rank << std::endl;

    t_condLatChol = get_time(0.0);

	double gflops_factorize = solver->factorize(ia, ja, a, dummy_time_1);
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

  	double gflops_solve = solver->solve(ia, ja, a, x, b, nrhs, dummy_time_1, dummy_time_2);
    //double gflops_solve = solver->solve(x, b, nrhs);

    t_condLatSolve = get_time(t_condLatSolve);

#ifdef PRINT_MSG
  	//printf("flops solve:     %f\n", flops_solve);
	printf("Residual norm: %e\n", solver->residualNorm(x, b));
	printf("Residual norm normalized: %e\n", solver->residualNormNormalized(x, b));
#endif

#ifdef PRINT_TIMES
	printf("BTA factorise time: %lg\n",t_condLatChol);
  	printf("BTA solve     time: %lg\n",t_condLatSolve);
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

void BTASolver::factorize_solve_w_constr(SpMat& Q, Vect& rhs, const MatrixXd& Dxy, double &log_det, Vect& sol, MatrixXd& V){

#ifdef PRINT_MSG
    std::cout << "in BTA FACTORIZE_SOLVE_W_CONSTR()." << std::endl;
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
    std::cout << "Calling solver = new BTA() now" << std::endl;
#endif

#ifdef PRINT_MSG
    printf("Calling BTA solver in BTA factorize_solver now.\n");
#endif

    double t_factorise = get_time(0.0);
    //solver->solve_equation(GR);
    double gflops_factorize = solver->factorize(ia, ja, a, dummy_time_1);
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
    double gflops_solve = solver->solve(ia, ja, a, x, b, nrhs, dummy_time_1, dummy_time_2);
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
    printf("BTA factorise time: %lg\n",t_factorise);
    printf("BTA solve     time: %lg\n",t_solve);
#endif

    delete[] ia;
    delete[] ja;
    delete[] a;
    delete[] x;
    delete[] b;

} // end factorize solve with constraints


// IMPLEMENT IN A WAY SUCH THAT FACTORISATION WILL BE PERFORMED AGAIN
// FOR NOW: cannot rely on factorisation to be there.
void BTASolver::selected_inversion_diag(SpMat& Q, Vect& inv_diag) {

#ifdef PRINT_MSG
    std::cout << "in BTA SELECTED_INVERSION()." << std::endl;
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
    double gflops_factorize = solver->factorize(ia, ja, a, dummy_time_1);
    t_factorise = get_time(t_factorise);

#ifdef PRINT_TIMES
    printf("BTA factorise time: %lg\n",t_factorise);
#endif

    t_inv = get_time(0.0);
    double gflops_inv = solver->BTAdiag(ia, ja, a, invDiag);
    t_inv = get_time(t_inv);

#ifdef PRINT_MSG
    printf("gflops factorise:      %f\n", gflops_factorize);
    printf("gflops inv      :      %f\n", gflops_inv);
#endif

#ifdef PRINT_TIMES
    printf("BTA factorise time: %lg\n",t_factorise);
    printf("BTA sel inv time  : %lg\n",t_inv);
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


void BTASolver::selected_inversion_diag_w_constr(SpMat& Q, const MatrixXd& D, Vect& inv_diag, MatrixXd& V){

    int nrhs = D.rows();

#ifdef PRINT_MSG
    std::cout << "in BTA SELECTED_INVERSION_W_CONSTR()." << std::endl;
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
    double flops_factorize = solver->factorize(ia, ja, a, dummy_time_1);
    t_factorise = get_time(t_factorise);

#ifdef PRINT_TIMES
    printf("BTA factorise time: %lg\n",t_factorise);
#endif

    t_solve = get_time(0.0); 
    double flops_solve = solver->solve(ia, ja, a, x, b, nrhs, dummy_time_1, dummy_time_2);
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
    double flops_inv = solver->BTAdiag(ia, ja, a, invDiag);
    t_inv = get_time(t_inv);

#ifdef PRINT_MSG
    printf("flops factorise:      %f\n", flops_factorize);
    printf("flops solve    :      %f\n", flops_solve);
    printf("flops inv      :      %f\n", flops_inv);
#endif

#ifdef PRINT_TIMES
    printf("BTA factorise time: %lg\n",t_factorise);
    printf("BTA solve time    : %lg\n",t_solve);
    printf("BTA sel inv time  : %lg\n",t_inv);
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


void BTASolver::selected_inversion_full(SpMat& Q, SpMat& Qinv) {

#ifdef PRINT_MSG
    std::cout << "in BTA SELECTED_INVERSION_FULL()." << std::endl;
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
    double* a  = new double [nnz];

    //double* invDiag  = new double[n];
    double* inva = new double [nnz];

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
    double gflops_factorize = solver->factorize(ia, ja, a, dummy_time_1);
    t_factorise = get_time(t_factorise);

#ifdef PRINT_TIMES
    printf("BTA factorise time: %lg\n",t_factorise);
#endif

    t_inv = get_time(0.0);
    double gflops_inv = solver->BTAselInv(ia, ja, a, inva);
    t_inv = get_time(t_inv);

#ifdef PRINT_MSG
    printf("gflops factorise:      %f\n", gflops_factorize);
    printf("gflops inv      :      %f\n", gflops_inv);
#endif

#ifdef PRINT_TIMES
    printf("BTA factorise time: %lg\n",t_factorise);
    printf("BTA sel inv time  : %lg\n",t_inv);
#endif

    SpMat Qinv_lower = Eigen::Map<Eigen::SparseMatrix<double> >(n,n,nnz,Q_lower.outerIndexPtr(), // read-write
                               Q_lower.innerIndexPtr(),inva);

    // TODO: more efficient way to do this?
    Qinv = Qinv_lower.selfadjointView<Lower>();

    // free memory
    delete[] ia;
    delete[] ja;
    delete[] a;
    delete[] inva;

} // end selected_inversion_full


void BTASolver::selected_inversion_full_w_constr(SpMat& Q, const MatrixXd& D, SpMat& Qinv, MatrixXd& V){
    std::cout << "Placeholder selected_inversion_fullTakInv_w_constr() doesnt exist for BTA solver yet." << std::endl;
    exit(1);
}


void BTASolver::compute_full_inverse(SpMat& Q, MatrixXd& Qinv) {

#ifdef PRINT_MSG
    std::cout << "MPI rank : " << MPI_rank << ", in BTA FACTORIZE_SOLVE()." << std::endl;   
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
    std::cout << "calling solver = new BTA now" << std::endl;
#endif

    //solver = new BTA<double>(ia, ja, a, ns_t, nt_t, nb_t);
    //solver = new BTA<double>(ns_t, nt_t, nb_t);


#ifdef PRINT_MSG
    printf("Calling BTA solver in BTA factorize_solver now.\n");
#endif

       // std::cout << "BTA factorize solve, nb = " << nb_t << ", MPI rank : " << MPI_rank << ", tid : " << omp_get_thread_num() << ", GPU rank : " << GPU_rank << std::endl;

    double t_condLatChol = get_time(0.0);

    double gflops_factorize = solver->factorize(ia, ja, a, dummy_time_1);
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

    // set rhs to identity
    int nrhs = n;
    int n2 = n*n;

    MatrixXd IdMat(n,n);
    IdMat.setIdentity();


    double* b      = new double[n2];
    double* x      = new double[n2];

    // assign b to correct format
    for (i = 0; i < n2; i++){
        b[i] = IdMat.data()[i];
        //printf("%f\n", b[i]);
    }

    double t_condLatSolve = get_time(0.0);

    double gflops_solve = solver->solve(ia, ja, a, x, b, nrhs, dummy_time_1, dummy_time_2);
    //double gflops_solve = solver->solve(x, b, nrhs);

    t_condLatSolve = get_time(t_condLatSolve);

#ifdef PRINT_MSG
    //printf("flops solve:     %f\n", flops_solve);
    printf("Residual norm: %e\n", solver->residualNorm(x, b));
    printf("Residual norm normalized: %e\n", solver->residualNormNormalized(x, b));
#endif

#ifdef PRINT_TIMES
    printf("BTA factorise time: %lg\n",t_condLatChol);
    printf("BTA solve     time: %lg\n",t_condLatSolve);
#endif

    //std::cout << "In factorize_solve. hostname : " << processor_name << ", MPI_rank : " << MPI_rank << ", GPU rank : " << GPU_rank << ", time Chol : " << t_condLatChol << ", time Solve : " << t_condLatSolve << std::endl;

    // assign b to correct format
    for (i = 0; i < n2; i++){
        Qinv.data()[i] = x[i];
    }   


    delete[] ia;
    delete[] ja;
    delete[] a;

    delete[] x;
    delete[] b;


}


BTASolver::~BTASolver(){
    //std::cout << "Derived destructor called." << std::endl;
    delete solver;

}




